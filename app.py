from flask import Flask, request, jsonify
from flask_cors import CORS
from movie_recommender import MovieRecommender, load_data
import os
from functools import lru_cache
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address

app = Flask(__name__)
CORS(app)

# Configuration du rate limiter
limiter = Limiter(
    app=app,
    key_func=get_remote_address,
    default_limits=["200 per day", "50 per hour"]
)

# Chargement des données et initialisation du recommender
print("Loading data...")
movies_data, feature_matrix = load_data()
recommender = MovieRecommender(movies_data, feature_matrix)
print("Data loaded successfully!")

# Cache pour la recherche
@lru_cache(maxsize=1000)
def search_movies(query):
    return movies_data[movies_data['title'].str.lower().str.contains(query.lower())]['title'].tolist()[:10]

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