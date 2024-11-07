import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity


# User ratings matrix
ratings_data = {
    'User 1': [5, 3, 4, 0],
    'User 2': [4, 0, 0, 2],
    'User 3': [0, 5, 4, 3]
}
ratings_matrix = pd.DataFrame(ratings_data, index=['Movie A', 'Movie B', 'Movie C', 'Movie D']).T

# Item features (one-hot encoding for genres)
item_features = np.array([
    [1, 0, 0, 1, 0],  # Movie A: Action, Adventure
    [0, 1, 1, 0, 0],  # Movie B: Comedy, Romance
    [1, 0, 0, 1, 0],  # Movie C: Action, Adventure
    [0, 0, 1, 0, 1]   # Movie D: Romance, Drama
])

# User preferences example (User 1 preferences)
user_preferences = np.array([1, 0, 0, 1, 0])  # User 1 prefers Action and Adventure

# Convert the ratings matrix to a NumPy array
ratings_array = ratings_matrix.values

# ---------------------------------------------

# Calculate user-user similarity using cosine similarity
user_similarity = cosine_similarity(ratings_array)

# Function to get recommendations based on collaborative filtering
def collaborative_filtering(user_index, ratings, user_similarity):
    similar_users = list(enumerate(user_similarity[user_index]))
    similar_users = sorted(similar_users, key=lambda x: x[1], reverse=True)
    
    recommendations = np.zeros(ratings.shape[1])  # Initialize recommendation scores

    for i, (other_user_index, similarity_score) in enumerate(similar_users):
        if other_user_index != user_index:  # Exclude self
            recommendations += similarity_score * ratings[other_user_index]

    return recommendations

# Get recommendations for User 1 (index 0)
collab_recommendations = collaborative_filtering(0, ratings_array, user_similarity)

# ---------------------------------------------

# Calculate content-based similarities
content_similarities = cosine_similarity(item_features)

# Function to get recommendations based on content filtering
def content_based_filtering(user_preferences, item_features):
    return cosine_similarity(user_preferences.reshape(1, -1), item_features).flatten()

# Get content-based recommendations for User 1
content_recommendations = content_based_filtering(user_preferences, item_features)

# ---------------------------------------------

# Combine recommendations (weighted average)
def hybrid_recommendation(collab_scores, content_scores, weights):
    return weights[0] * collab_scores + weights[1] * content_scores

# Set weights for collaborative and content-based recommendations
weights = [0.5, 0.5]

# Generate final scores
final_scores = hybrid_recommendation(collab_recommendations, content_recommendations, weights)

# Get recommended item indices (sort by score in descending order)
recommended_item_indices = np.argsort(final_scores)[::-1]

# Output recommended items
recommended_items = ratings_matrix.columns[recommended_item_indices]
print("Recommended items for User 1:", recommended_items)
