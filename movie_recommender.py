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
    
    def get_recommendations(self, movie_title, user_id, n_recommendations=5):
        """
        Retourne les recommandations en tenant compte des préférences de l'utilisateur
        """
        # Recherche du film de manière plus robuste
        movie_mask = self.movies_data['title'].str.lower() == movie_title.lower()
        if not movie_mask.any():
            return "Film non trouvé dans la base de données"
        
        # Obtenir l'index du film
        movie_idx = movie_mask.idxmax()
        
        # Conversion en array dense pour le film de référence
        movie_features = sparse.csr_matrix(self.feature_matrix.iloc[movie_idx]).toarray()
        
        # Calculer les similarités par lots pour économiser la mémoire
        batch_size = 1000
        n_samples = len(self.feature_matrix)
        similarities = np.zeros(n_samples)
        
        for i in range(0, n_samples, batch_size):
            end = min(i + batch_size, n_samples)
            # Conversion en array dense pour le batch
            batch = sparse.csr_matrix(self.feature_matrix.iloc[i:end]).toarray()
            similarities[i:end] = cosine_similarity(movie_features, batch)[0]
        
        # Ajuster les scores en fonction des préférences utilisateur
        adjusted_scores = self._adjust_scores(similarities, user_id)
        
        # Obtenir les indices des films les plus similaires
        top_indices = np.argpartition(adjusted_scores, -n_recommendations-1)[-n_recommendations-1:]
        top_indices = top_indices[np.argsort(adjusted_scores[top_indices])][::-1]
        
        # Filtrer les films déjà vus et le film actuel
        seen_movies = self.user_preferences[user_id]['liked'] | self.user_preferences[user_id]['disliked']
        current_movie = self.movies_data.iloc[movie_idx]['title']
        
        filtered_indices = []
        for idx in top_indices:
            movie_title = self.movies_data.iloc[idx]['title']
            if movie_title not in seen_movies and movie_title != current_movie:
                filtered_indices.append(idx)
            if len(filtered_indices) >= n_recommendations:
                break
        
        # Créer un DataFrame avec les films similaires
        recommendations = pd.DataFrame({
            'title': self.movies_data.iloc[filtered_indices]['title'],
            'similarity_score': adjusted_scores[filtered_indices],
            'genres': self.movies_data.iloc[filtered_indices]['genres'],
            'rating': self.movies_data.iloc[filtered_indices]['rating'],
            'poster_url': self.movies_data.iloc[filtered_indices]['poster_url'],
            'summary': self.movies_data.iloc[filtered_indices]['summary']
        })
        
        return recommendations
    
    def add_feedback(self, user_id, movie_title, liked=True):
        if liked:
            self.user_preferences[user_id]['liked'].add(movie_title)
        else:
            self.user_preferences[user_id]['disliked'].add(movie_title)
        
        self._update_genre_weights(user_id)
    
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