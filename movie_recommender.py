import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from collections import defaultdict
import joblib
from scipy import sparse

def load_data():
    # Chargement du modèle et des données
    model_data = joblib.load('movie_recommender_model.joblib')
    return model_data['movies_data'], model_data['feature_matrix']

class MovieRecommender:
    def __init__(self, movies_data, feature_matrix):
        self.movies_data = movies_data.reset_index(drop=True)
        self.feature_matrix = feature_matrix.reset_index(drop=True)
        self.user_preferences = defaultdict(lambda: {'liked': set(), 'disliked': set()})
        self.genre_weights = defaultdict(lambda: 1.0)
    
    def get_recommendations(self, movie_title, user_id):
        try:
            # Trouver l'index du film
            movie_idx = self.movies_data[self.movies_data['title'] == movie_title].index[0]
            base_similarities = self.compute_similarity(movie_idx)
            
            # Si l'utilisateur a des préférences, les prendre en compte
            if user_id in self.user_profiles and len(self.user_preferences.get(user_id, {})) > 0:
                user_profile = self.user_profiles[user_id]
                user_similarities = self.compute_user_similarity(user_profile)
                
                # Combiner les similarités basées sur le film et sur l'utilisateur
                combined_similarities = (base_similarities + user_similarities) / 2
            else:
                combined_similarities = base_similarities
            
            # Obtenir les meilleurs films
            top_indices = combined_similarities.argsort()[-11:][::-1]
            
            # Exclure le film de référence
            top_indices = top_indices[top_indices != movie_idx][:10]
            
            recommendations = self.movies_data.iloc[top_indices].copy()
            recommendations['similarity_score'] = combined_similarities[top_indices]
            
            return recommendations
            
        except Exception as e:
            print(f"Error in get_recommendations: {str(e)}")
            return str(e)

    def compute_user_similarity(self, user_profile):
        # Calculer la similarité entre le profil utilisateur et tous les films
        similarities = np.dot(self.feature_matrix, user_profile)
        similarities = (similarities + 1) / 2  # Normaliser entre 0 et 1
        return similarities

    def add_feedback(self, user_id, movie_title, liked):
        # Stocker le feedback
        if user_id not in self.user_preferences:
            self.user_preferences[user_id] = {}
        
        # Mettre à jour les préférences avec un poids plus important
        weight = 1.0 if liked else -1.0
        self.user_preferences[user_id][movie_title] = weight
        
        # Mettre à jour le profil utilisateur
        movie_idx = self.movies_data[self.movies_data['title'] == movie_title].index
        if len(movie_idx) > 0:
            movie_features = self.feature_matrix[movie_idx[0]]
            
            if user_id not in self.user_profiles:
                self.user_profiles[user_id] = np.zeros_like(movie_features)
            
            # Mettre à jour le profil avec les nouvelles préférences
            self.user_profiles[user_id] = (
                self.user_profiles[user_id] + (weight * movie_features)
            ) / 2  # Moyenne pondérée avec le profil existant
        
        return True
    
    def _update_genre_weights(self, user_id):
        liked_movies = self.user_preferences[user_id]['liked']
        disliked_movies = self.user_preferences[user_id]['disliked']
        
        self.genre_weights = defaultdict(lambda: 1.0)
        
        for movie in liked_movies:
            movie_mask = self.movies_data['title'] == movie
            if movie_mask.any():
                genres = self.movies_data[movie_mask]['genres'].iloc[0]
                for genre in genres:
                    self.genre_weights[genre] *= 1.2
        
        for movie in disliked_movies:
            movie_mask = self.movies_data['title'] == movie
            if movie_mask.any():
                genres = self.movies_data[movie_mask]['genres'].iloc[0]
                for genre in genres:
                    self.genre_weights[genre] *= 0.8
    
    def _adjust_scores(self, similarities, user_id):
        adjusted_scores = similarities.copy()
        
        for i, movie in enumerate(self.movies_data['title']):
            genres = self.movies_data.iloc[i]['genres']
            genre_weight = np.mean([self.genre_weights[genre] for genre in genres]) if genres else 1.0
            adjusted_scores[i] *= genre_weight
        
        return adjusted_scores 