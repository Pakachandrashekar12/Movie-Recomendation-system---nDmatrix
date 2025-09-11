import pandas as pd
import pickle
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Sample movie dataset
data = {
    'movie_id': [1, 2, 3, 4, 5],
    'title': ['The Matrix', 'John Wick', 'Avengers', 'Interstellar', 'Inception'],
    'genre': ['Action Sci-Fi', 'Action Thriller', 'Action Adventure', 'Sci-Fi Drama', 'Sci-Fi Thriller']
}

# Convert to DataFrame
movies = pd.DataFrame(data)

# Convert genres into feature vectors
vectorizer = CountVectorizer()
genre_matrix = vectorizer.fit_transform(movies['genre'])

# Compute similarity between movies
cosine_sim = cosine_similarity(genre_matrix, genre_matrix)

# Save files as pickle
pickle.dump(movies.to_dict(), open('src/movie_dict.pkl', 'wb'))
pickle.dump(cosine_sim, open('src/similarity.pkl', 'wb'))

print("✅ Training complete! Files saved in src/: movie_dict.pkl & similarity.pkl")
