import torch
import torch.nn as nn
import numpy as np
from scipy.sparse import csr_matrix, lil_matrix
from scipy.sparse.linalg import eigsh
from scipy.linalg import inv
from copy import deepcopy
from datetime import datetime, timedelta
from recbole.utils import InputType
from recbole.model.abstract_recommender import GeneralRecommender


class DynamicEASEr(GeneralRecommender):
    """
    Dynamic EASEr (Embarrassingly Shallow Auto-Encoders for Dynamic Collaborative Filtering)
    
    Reference:
        Steck, Harald. "Embarrassingly shallow autoencoders for sparse data." WWW 2019.
        Extended to support dynamic updates.
    """
    input_type = InputType.POINTWISE
    
    def __init__(self, config, dataset):
        super(DynamicEASEr, self).__init__(config, dataset)
        
        # Load dataset info
        self.n_users = dataset.user_num
        self.n_items = dataset.item_num
        
        # Load parameters
        self.l2_reg = config['l2_reg'] if 'l2_reg' in config else 500.0
        self.update_interval = config['update_interval'] if 'update_interval' in config else 60  # minutes
        self.rank_method = config['rank_method'] if 'rank_method' in config else 'user_bound'
        
        # Initialize model matrices
        self.X = None  # User-item interaction matrix (lil_matrix for efficient updates)
        self.G = None  # Gramian matrix
        self.B = None  # Item-item similarity matrix (the model)

        self.is_fit = False

        # Create initial matrices
        self._create_interaction_matrix(dataset)
        
    def _create_interaction_matrix(self, dataset):
        """
        Create the initial user-item interaction matrix from the dataset.
        """
        # Create sparse user-item matrix from the dataset
        rows, cols = [], []
        timestamps = []
        
        for interaction in dataset:
            user_idx = interaction[self.USER_ID].item()
            item_idx = interaction[self.ITEM_ID].item()
            # Check if timestamp is available in the dataset
            if hasattr(interaction, 'timestamp'):
                timestamp = interaction['timestamp'].item()
            else:
                timestamp = 0  # Default timestamp if not provided
                
            rows.append(user_idx)
            cols.append(item_idx)
            timestamps.append(timestamp)
            
        # Create the interaction matrix
        data = np.ones(len(rows))
        self.X = lil_matrix((self.n_users, self.n_items), dtype=np.float64)
        
        # Fill the matrix with ones for each user-item interaction
        for r, c in zip(rows, cols):
            self.X[r, c] = 1.0
    
    def compute_gramian(self, X):
        """
        Compute Gramian matrix for user-item matrix X.
        """
        X_csr = X.tocsr() if not isinstance(X, csr_matrix) else X
        G = (X_csr.T @ X_csr).toarray()
        return G
    
    def add_diagonal(self, G, l2):
        """
        Add L2 regularization to the Gramian matrix.
        """
        return G + l2 * np.eye(G.shape[0])
    
    def initial_train(self):
        """
        Train the initial EASEr model.
        """
        if self.X is None:
            raise ValueError("Interaction matrix not initialized.")
            
        # Compute Gramian matrix
        self.G = self.compute_gramian(self.X)
        
        # Add L2 regularization
        G_with_l2 = self.add_diagonal(self.G, self.l2_reg)
        
        # Compute the inverse of the Gramian matrix
        self.B = inv(G_with_l2)
        
        # Set training flag
        self.is_fit = True
    
    def dyngram(self, new_interactions):
        """
        Update the Gramian matrix incrementally with new interactions.
        
        Args:
            new_interactions: List of tuples (user_id, item_id)
        """
        # Placeholder for row and column indices for G_delta
        r, c = [], []
        
        # For every new interaction
        for user_id, item_id in new_interactions:
            # For every item already seen by this user
            for seen_item in self.X[user_id, :].nonzero()[1]:
                # Update co-occurrence at (i,j)
                r.extend([item_id, seen_item])
                c.extend([seen_item, item_id])
            
            # Update occurrence for item at (i,i)
            r.append(item_id)
            c.append(item_id)
            
            # Update the interaction matrix
            self.X[user_id, item_id] = 1.0
            
        # Create sparse matrix for Gramian diff
        G_diff = csr_matrix((np.ones_like(r, dtype=np.float64), (r, c)), 
                           shape=(self.n_items, self.n_items))
        
        return G_diff
    
    def dynamic_update(self, new_interactions):
        """
        Perform dynamic update on the model.
        
        Args:
            new_interactions: List of tuples (user_id, item_id)
        """
        if not self.is_fit:
            raise ValueError("Initial training must be completed before dynamic updates.")
        
        # Compute Gramian difference
        G_diff = self.dyngram(new_interactions)
        
        # Extract non-zero rows from Gramian diff
        nnz = list(set(G_diff.nonzero()[0]))
        
        # Only update if at least one non-zero entry in the Gramian diff
        if len(nnz) > 0:
            # Compute rank for the update
            n_users = len(set([u for u, _ in new_interactions]))
            n_items = len(set([i for _, i in new_interactions]))
            
            # Determine rank based on specified method
            if self.rank_method == 'exact':
                k = np.linalg.matrix_rank(G_diff[nnz, :][:, nnz].todense(), hermitian=True)
            elif self.rank_method == 'user_bound':
                k = 2 * n_users
            elif self.rank_method == 'item_bound':
                k = 2 * n_items
            else:
                k = min(n_users, n_items)  # Default fallback
                
            # Ensure k is not greater than the number of non-zero elements
            k = min(k, len(nnz))
            
            if k > 0:
                # Compute eigendecomposition
                vals, vecs = eigsh(G_diff, k=k)
                vals, vecs = deepcopy(vals.real), deepcopy(vecs.real)
                
                # Update the model through the Woodbury Identity
                VAinv = vecs.T @ self.B
                self.B -= (self.B @ vecs) @ inv(np.diag(1.0 / vals) + VAinv @ vecs) @ VAinv
    
    def forward(self, user, item):
        """
        Predict the score of a user-item pair.
        """
        if not self.is_fit:
            self.initial_train()
            
        # Compute the recommendation score
        # Since B is the item-item similarity matrix, we need to exclude the diagonal
        B_no_diag = self.B.copy()
        np.fill_diagonal(B_no_diag, 0)
        
        # Create a one-hot vector for the user's history
        user_history = torch.zeros(self.n_items, device=self.device)
        if isinstance(user, torch.Tensor) and user.dim() == 0:
            user = user.item()
            
        # Get the user's history from the interaction matrix
        if isinstance(user, int):
            user_viewed_items = self.X[user].nonzero()[1]
            user_history[user_viewed_items] = 1.0
        else:
            # Handle batch processing if needed
            raise NotImplementedError("Batch processing not implemented for this model")
            
        # Compute scores using the similarity matrix
        scores = torch.matmul(torch.tensor(B_no_diag, device=self.device), 
                             user_history.unsqueeze(1)).squeeze()
        
        # Return the score for the specific item
        if isinstance(item, torch.Tensor) and item.dim() == 0:
            return scores[item.item()]
        elif isinstance(item, int):
            return scores[item]
        else:
            # Handle batch processing if needed
            raise NotImplementedError("Batch processing not implemented for this model")
    
    def calculate_loss(self, interaction, update=True):
        """
        Calculate the loss for optimization.
        Note: EASEr is a closed-form solution, so no backpropagation is needed.
        """
        if update:
            # Collect new interactions since last update
            new_interactions = []
            user = interaction[self.USER_ID]
            item = interaction[self.ITEM_ID]
            
            # Process each batch interaction
            for u, i in zip(user, item):
                new_interactions.append((u.item(), i.item()))
                
            # Perform dynamic update
            self.dynamic_update(new_interactions)
            
        # EASEr doesn't require gradient-based optimization, return dummy loss
        return torch.tensor(0.0, device=self.device, requires_grad=True)
    
    def predict(self, interaction):
        """
        Predict the rating/score of the interaction.
        """
        user = interaction[self.USER_ID]
        item = interaction[self.ITEM_ID]
        
        # Handle batch prediction
        if user.dim() > 0:
            scores = []
            for u, i in zip(user, item):
                scores.append(self.forward(u, i))
            return torch.tensor(scores, device=self.device)
        else:
            return self.forward(user, item)
    
    def add_items(self, new_item_count):
        """
        Expand the model to accommodate new items.
        
        Args:
            new_item_count (int): Number of new items to add
            
        Returns:
            tuple: (starting_item_id, ending_item_id) range of new item IDs
        """
        if new_item_count <= 0:
            return None
            
        # Get the current dimensions
        start_item_id = self.n_items
        
        # Resize the interaction matrix
        new_X = lil_matrix((self.n_users, self.n_items + new_item_count), dtype=np.float64)
        new_X[:, :self.n_items] = self.X
        self.X = new_X
        
        # If the model is already trained, we need to expand the Gramian and B matrices
        if self.is_fit:
            # Expand Gramian matrix
            new_G = np.zeros((self.n_items + new_item_count, self.n_items + new_item_count))
            new_G[:self.n_items, :self.n_items] = self.G
            self.G = new_G
            
            # Expand B matrix with appropriate regularization
            # We create diagonal entries for new items with the same regularization
            new_B = np.zeros((self.n_items + new_item_count, self.n_items + new_item_count))
            new_B[:self.n_items, :self.n_items] = self.B
            
            # Initialize new items' diagonal with inverse of L2 regularization
            # This is because the initial Gramian for these items would be 0 + l2*I
            for i in range(self.n_items, self.n_items + new_item_count):
                new_B[i, i] = 1.0 / self.l2_reg
                
            self.B = new_B
        
        # Update item count
        self.n_items += new_item_count
        
        return (start_item_id, self.n_items - 1)
    
    def remove_user(self, user_id):
        """
        Remove a user from the model.
        Note: This doesn't actually change the model parameters but
        marks the user's interactions as removed for future updates.
        
        Args:
            user_id (int): The ID of the user to remove
            
        Returns:
            bool: True if the user existed and was removed, False otherwise
        """
        if user_id >= self.n_users:
            return False
            
        # Get existing interactions
        user_interactions = self.X[user_id].nonzero()[1]
        if len(user_interactions) == 0:
            return False
            
        # Clear all interactions for this user
        self.X[user_id, :] = 0
        
        # We don't immediately update the model because it would be computationally expensive
        # Instead, the next dynamic update will account for these changes
        
        return True
    
    def add_user(self, user_id, item_ids, update_model=True):
        """
        Add a new user with their initial interactions to the model.
        
        Args:
            user_id (int): The ID of the new user
            item_ids (list): List of item IDs that the user has interacted with
            update_model (bool): Whether to update the model immediately (default: True)
            
        Returns:
            bool: True if the user was added successfully, False otherwise
        """
        # Check if user_id is valid (within bounds)
        if user_id >= self.n_users:
            # Need to resize the interaction matrix to accommodate the new user
            new_rows = user_id - self.n_users + 1
            new_X = lil_matrix((self.n_users + new_rows, self.n_items), dtype=np.float64)
            
            # Copy existing data
            new_X[:self.n_users, :] = self.X
            self.X = new_X
            self.n_users = user_id + 1
            print(f"Expanded user matrix to accommodate new user. New user count: {self.n_users}")
        
        # Check if the user already exists with interactions
        user_interactions = self.X[user_id].nonzero()[1]
        if len(user_interactions) > 0:
            print(f"User {user_id} already exists with {len(user_interactions)} interactions")
        
        # Add the new interactions
        new_interactions = []
        for item_id in item_ids:
            # Validate item ID
            if item_id >= self.n_items:
                print(f"Warning: Item {item_id} is out of range and will be ignored")
                continue
                
            # Add to interactions list for model update
            new_interactions.append((user_id, item_id))
            
            # Update the interaction matrix directly
            self.X[user_id, item_id] = 1.0
            
        # Update the model if requested
        if update_model and self.is_fit and new_interactions:
            self.dynamic_update(new_interactions)
            print(f"Model updated with {len(new_interactions)} new interactions for user {user_id}")
            
        return len(new_interactions) > 0
    
    def full_sort_predict(self, interaction):
        """
        Predict scores for all items for each user in the batch.
        """
        user = interaction[self.USER_ID]

        # Ensure initial training is complete
        if not self.is_fit:
            self.initial_train()

        # Compute the item-item similarity matrix without diagonal
        B_no_diag = self.B.copy()
        np.fill_diagonal(B_no_diag, 0)
        B_tensor = torch.tensor(B_no_diag, device=self.device)
        
        # Process each user
        if user.dim() == 0:
            # Single user case
            user_idx = user.item()
            user_history = torch.zeros(self.n_items, device=self.device)
            user_viewed_items = self.X[user_idx].nonzero()[1]
            user_history[user_viewed_items] = 1.0
            
            # Compute scores for all items
            scores = torch.matmul(B_tensor, user_history.unsqueeze(1)).squeeze()
            return scores.unsqueeze(0)  # Add batch dimension
        else:
            # Batch processing
            batch_size = len(user)
            scores = torch.zeros(batch_size, self.n_items, device=self.device)
            
            for idx, u in enumerate(user):
                user_idx = u.item()
                user_history = torch.zeros(self.n_items, device=self.device)
                user_viewed_items = self.X[user_idx].nonzero()[1]
                user_history[user_viewed_items] = 1.0
                
                # Compute scores for all items
                scores[idx] = torch.matmul(B_tensor, user_history.unsqueeze(1)).squeeze()
                
            return scores