import os
import torch
import recbole
import numpy as np
import scipy.sparse as sp

from tqdm import tqdm
from recbole.model.abstract_recommender import ContextRecommender
from flurs.recommender import FMRecommender
from flurs.data.entity import User, Item, Event, BaseActor


class IncrementalFM(FMRecommender, ContextRecommender):
    """Incremental Factorization Machines adapted to RecBole from FluRS.

    Reference
    ----------
    * Kitazawa, T. (2016, July). Incremental Factorization Machines for
    Persistently Cold-starting Online Item Recommendation (2016)
    """

    incremental: bool = True

    def __init__(self, config, dataset, p=None, k=40, l2_reg_w0=2.0, l2_reg_w=8.0, l2_reg_V=16.0, learn_rate=0.004, incremental_fit=False, seed=None):
        self.incremental_fit = incremental_fit
        self.seed = seed
        self.config = config
        self.dataset = dataset

        if seed:
            np.random.seed(seed)

        ContextRecommender.__init__(self, config, dataset)
        super(FMRecommender, self).__init__(
            p=p, k=k, l2_reg_w0=l2_reg_w0, l2_reg_w=l2_reg_w, l2_reg_V=l2_reg_V, learn_rate=learn_rate
        )

        super().initialize(use_index=True)

        self.fake_loss = torch.nn.Parameter(torch.zeros(1))

        self.other_parameter_name = [
            "p",
            "k",
            "l2_reg_w0",
            "l2_reg_w",
            "l2_reg_V",
            "learn_rate",
            "incremental_fit",
            "use_index",
            "w0",
            "w",
            "V",
            "prev_w0",
            "prev_w",
            "prev_V"
        ]


    def register_users(self, users: list[User]):
        """
        Register multiple users at once.

        Parameters:
        users (list of User): List of User instances to register.
        """
        if not self.use_index:
            return

        self.users.update({user.index: {"known_items": set()}
                          for user in users})
        n_users = len(users)
        n_user = self.n_user - 1

        self.n_user += n_users
        self.p += n_users

        # Insert new users' rows for the parameters
        self.w = np.insert(self.w, n_user + 1, np.zeros(n_users), axis=0)
        self.prev_w = np.insert(self.prev_w, n_user + 1,
                                np.zeros(n_users), axis=0)

        rand_rows = np.random.normal(0.0, 0.1, (n_users, self.k))
        self.V = np.insert(self.V, n_user + 1, rand_rows, axis=0)
        self.prev_V = np.insert(self.prev_V, n_user + 1, rand_rows, axis=0)

    def register_items(self, items: list[Item]):
        """
        Register multiple items at once.

        Parameters:
        items (list of Item): List of Item instances to register.
        """
        self.items.update({item.index: {} for item in items})
        n_items = len(items)
        n_item = self.n_item - 1

        self.n_item += n_items

        # Update the item matrix for all items
        i_vecs = self._encode_actors(
            items, dim=self.n_item, index=self.use_index, feature=True, vertical=True)
        sp_i_vecs = sp.csr_matrix(np.hstack(i_vecs))

        if self.i_mat.size == 0:
            self.i_mat = sp_i_vecs
        elif self.use_index:
            self.i_mat = sp.vstack(
                (self.i_mat[:n_item], np.zeros(
                    (n_items, n_item)), self.i_mat[n_item:])
            )
            self.i_mat = sp.csr_matrix(sp.hstack((self.i_mat, sp_i_vecs)))
        else:
            self.i_mat = sp.csr_matrix(sp.hstack((self.i_mat, sp_i_vecs)))

        if self.use_index:
            # Insert new items' rows for the parameters
            self.w = np.insert(self.w, n_item, np.zeros(n_items), axis=0)
            self.prev_w = np.insert(
                self.prev_w, n_item, np.zeros(n_items), axis=0)

            rand_rows = np.random.normal(0.0, 0.1, (n_items, self.k))
            self.V = np.insert(self.V, n_item, rand_rows, axis=0)
            self.prev_V = np.insert(self.prev_V, n_item, rand_rows, axis=0)
            self.p += n_items

    def train(self, mode: bool = True):
        if self.w.flatten().sum() > 0:
            return
        
        inter_data = self.dataset.inter_feat
        users = inter_data[self.USER_ID].values
        items = inter_data[self.ITEM_ID].values

        if self.dataset.user_feat:
            self.register_users([
                User(u, self.dataset.user_feat[u])
                for u in users
            ])
        else:
            self.register_users([
                User(u)
                for u in users
            ])

        if self.dataset.item_feat:
            self.register_items([
                Item(i, self.dataset.item_feat[i])
                for i in items
            ])
        else:
            self.register_items([
                Item(i)
                for i in items
            ])

        events = []
        for u, i, r in inter_data.itertuples(index=False):
            if r > 0:
                event = Event(User(u[0]), Item(i[0]), value=1.0)
                events.append(event)

        if self.incremental_fit:
            [
                self.update(e)
                for e in tqdm(events)
            ]
        else:
            self.fit_events(
                events,
                user_feature=bool(self.dataset.user_feat),
                item_feature=bool(self.dataset.item_feat)
            )
        super().train(mode)

    def calculate_loss(self, interaction):
        return torch.nn.Parameter(torch.zeros(1))

    def predict(self, interaction):
        user = interaction[self.USER_ID]
        item = interaction[self.ITEM_ID]
        user = user.cpu().numpy().astype(int)
        item = item.cpu().numpy().astype(int)
        result = []

        for index in range(len(user)):
            uid = user[index]
            iid = item[index]
            score = self.score(uid, iid)
            result.append(score)
        result = torch.from_numpy(np.array(result)).to(self.device)
        return result

    def full_sort_predict(self, interaction):
        user = interaction[self.USER_ID]
        user = user.cpu().numpy()

        result = torch.from_numpy(self.score(user)).to(self.device)

        return result

    def fit_events(self, events: list[Event], user_feature=True, item_feature=True):
        X = self._encode_events(
            events,
            user_feature=user_feature,
            item_feature=item_feature
        )

        assert X.shape[0] > 0, "feature vector has zero dimension"

        y = np.full(len(events), 1.0)  # implicit feedback

        self._fit_model(X, y)

    def score(self, user_idx, item_idx=None):
        """Predict the scores/ratings of a user for an item.

        Parameters
        ----------
        user_idx: int, required
            The index of the user for whom to perform score prediction.

        item_idx: int, optional, default: None
            The index of the item for that to perform score prediction.
            If None, scores for all known items will be returned.

        Returns
        -------
        res : A scalar or a Numpy array
            Relative scores that the user gives to the item or to all known items

        """
        if item_idx is None:
            return 1 - FMRecommender.score(self, User(user_idx), list(self.iid_map.values()))
        else:
            return 1 - FMRecommender.score(self, User(user_idx), [self.iid_map[item_idx]])[0]

    def _fit_model(self, X, y):
        """
        Fit the iFM model to a dataset.

        Parameters:
        X (numpy.ndarray): 2D array of embeddings (shape: n_samples x n_features)
        y (numpy.ndarray): 1D array of implicit feedback (shape: n_samples)
        """
        [
            self.update_model(x, value)
            for x, value in tqdm(zip(X, y), total=len(y))
        ]

    def _encode_actors(self, actors: list[BaseActor], dim=None, index=True, feature=True, vertical=False):
        if dim is None:
            dim = max(actor.index for actor in actors) + 1

        encoded_matrix = np.empty((len(actors), 0))

        if index:
            encoded_matrix = np.zeros((len(actors), dim))
            indices = np.array([actor.index for actor in actors])
            # one-hot encoding by index
            encoded_matrix[np.arange(len(actors)), indices] = 1.0

        if feature:
            feature_matrix = np.array([actor.feature for actor in actors])
            encoded_matrix = np.hstack((encoded_matrix, feature_matrix))

        return encoded_matrix if not vertical else np.array([encoded_matrix.T])

    def _encode_events(self, events: list[Event], user_feature=True, item_feature=True):
        user_encodings = self._encode_actors(
            [e.user for e in events],
            dim=self.n_user,
            index=self.use_index,
            feature=user_feature,
            vertical=False
        )
        item_encodings = self._encode_actors(
            [e.item for e in events],
            dim=self.n_item,
            index=self.use_index,
            feature=item_feature,
            vertical=False
        )
        contexts = np.array([event.context for event in events])

        return np.hstack((user_encodings, contexts, item_encodings))

if __name__ == '__main__':

    models_configs = [
        ('BPR',      'config/generic.yaml'),
        ('EASE',     'config/generic.yaml'),
        ('FM',       'config/generic.yaml'),
        ('LightGCN', 'config/generic.yaml'),
        ('MultiVAE', 'config/generic.yaml'),
        ('xDeepFM',  'config/generic.yaml'),
        ('WideDeep', 'config/generic.yaml'),
        ('Pop',      'config/once.yaml'),
        ('Random',   'config/once.yaml'),
    ]

    models, configs = zip(*models_configs)

    # os.system(f"""python run_recbole_group.py --[model_list]={','.join(models)} --[dataset]=[dataset] --[config_files]=[config_files] --[valid_latex]=[valid_latex] --[test_latex]=[test_latex]""")
    os.system(f"""python run_recbole_group.py --[model_list]={','.join(
        models)} --[dataset]=ml-100k --[config_files]={','.join(configs)}""")
