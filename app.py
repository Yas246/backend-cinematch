from flask import Flask, request, jsonify
from flask_cors import CORS
from movie_recommender import MovieRecommender, load_data
import os
from functools import lru_cache
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
import gc

app = Flask(__name__)
CORS(app)

# Configuration du rate limiter avec Redis
limiter = Limiter(
    app=app,
    key_func=get_remote_address,
    default_limits=["200 per day", "50 per hour"],
    storage_uri="memory://"  # Utilisez "redis://localhost:6379" en production
)

# Chargement des données avec gestion de mémoire optimisée
print("Loading data...")
try:
    movies_data, feature_matrix = load_data()
    # Forcer la collection des objets non utilisés
    gc.collect()
    print("Data loaded successfully!")
except Exception as e:
    print(f"Error loading data: {str(e)}")
    raise

# Initialisation du recommender
recommender = MovieRecommender(movies_data, feature_matrix)

# Cache pour la recherche avec taille limitée
@lru_cache(maxsize=100)
def search_movies(query):
    try:
        # Limiter les résultats pour économiser la mémoire
        return movies_data[movies_data['title'].str.lower().str.contains(query.lower())]['title'].head(10).tolist()
    except Exception as e:
        print(f"Search error in cache: {str(e)}")
        return []

# Configuration de Gunicorn pour la production
workers = int(os.getenv('GUNICORN_WORKERS', 1))
threads = int(os.getenv('GUNICORN_THREADS', 2))
timeout = int(os.getenv('GUNICORN_TIMEOUT', 30))

@app.route('/search')
@limiter.limit("20 per minute")
def search():
    query = request.args.get('query', '').lower()
    if len(query) < 2:
        return jsonify([])
    try:
        matching_movies = search_movies(query)
        return jsonify(matching_movies)
    except Exception as e:
        print(f"Search error: {str(e)}")
        return jsonify([])

@app.route('/movie_details')
@limiter.limit("30 per minute")
def movie_details():
    try:
        movie_title = request.args.get('movie')
        if not movie_title:
            return jsonify({'error': 'No movie title provided'})
        
        movie_mask = movies_data['title'] == movie_title
        if not movie_mask.any():
            return jsonify({'error': 'Movie not found'})
        
        movie = movies_data[movie_mask].iloc[0]
        
        return jsonify({
            'movie': {
                'title': movie['title'],
                'genres': movie['genres'],
                'rating': movie['rating'],
                'poster_url': movie['poster_url'],
                'summary': movie.get('summary', ''),
            }
        })
    except Exception as e:
        print(f"Movie details error: {str(e)}")
        return jsonify({'error': 'Internal server error'})

@app.route('/recommend')
@limiter.limit("20 per minute")
def recommend():
    try:
        movie_title = request.args.get('movie')
        user_id = request.args.get('user_id', 'default_user')
        
        if not movie_title:
            return jsonify({'error': 'No movie title provided'})
        
        recommendations = recommender.get_recommendations(movie_title, user_id)
        
        if isinstance(recommendations, str):
            return jsonify({'error': recommendations})
        
        return jsonify({
            'recommendations': recommendations.to_dict('records')
        })
    except Exception as e:
        print(f"Recommendation error: {str(e)}")
        return jsonify({'error': 'Internal server error'})

@app.route('/feedback', methods=['POST'])
@limiter.limit("30 per minute")
def feedback():
    try:
        data = request.json
        user_id = data.get('user_id', 'default_user')
        movie_title = data.get('movie_title')
        liked = data.get('liked', True)
        
        if not movie_title:
            return jsonify({'error': 'No movie title provided'})
        
        recommender.add_feedback(user_id, movie_title, liked)
        return jsonify({'success': True})
    except Exception as e:
        print(f"Feedback error: {str(e)}")
        return jsonify({'error': 'Internal server error'})

if __name__ == '__main__':
    port = os.getenv("PORT", 8000)
    app.run(host="0.0.0.0", port=port)