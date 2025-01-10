from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import pandas as pd
from movie_recommender import MovieRecommender, load_data
import os

app = Flask(__name__)
CORS(app)

# Chargement des données et initialisation du recommender
movies_data, feature_matrix = load_data()
recommender = MovieRecommender(movies_data, feature_matrix)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/search')
def search():
    query = request.args.get('query', '').lower()
    matching_movies = movies_data[movies_data['title'].str.lower().str.contains(query)]['title'].tolist()
    return jsonify(matching_movies[:10])

@app.route('/recommend')
def recommend():
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

@app.route('/feedback', methods=['POST'])
def feedback():
    data = request.json
    user_id = data.get('user_id', 'default_user')
    movie_title = data.get('movie_title')
    liked = data.get('liked', True)
    
    if not movie_title:
        return jsonify({'error': 'No movie title provided'})
    
    recommender.add_feedback(user_id, movie_title, liked)
    return jsonify({'success': True})

@app.route('/movie_details')
def movie_details():
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
            'summary': movie.get('summary', ''),  # Ajoutez d'autres champs si disponibles
        }
    })

if __name__ == '__main__':
    port = os.getenv("PORT", 5000)
    app.run(host="0.0.0.0", port=port)