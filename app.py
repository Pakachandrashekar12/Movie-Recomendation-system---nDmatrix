import streamlit as st
import pickle
import pandas as pd

# Load your trained model (example: similarity matrix or recommender function)
# Update path if needed
movies_dict = pickle.load(open('src/movie_dict.pkl', 'rb'))
similarity = pickle.load(open('src/similarity.pkl', 'rb'))
movies = pd.DataFrame(movies_dict)

def recommend(movie):
    movie_index = movies[movies['title'] == movie].index[0]
    distances = similarity[movie_index]
    movie_list = sorted(list(enumerate(distances)), reverse=True, key=lambda x: x[1])[1:6]
    recommended_movies = []
    for i in movie_list:
        recommended_movies.append(movies.iloc[i[0]].title)
    return recommended_movies

# Streamlit frontend
st.title("🎬 Movie Recommendation System")
st.write("Find movies similar to your favorite one!")

selected_movie = st.selectbox(
    "Type or select a movie from the dropdown",
    movies['title'].values
)

if st.button('Recommend'):
    recommendations = recommend(selected_movie)
    st.subheader("Recommended Movies:")
    for rec in recommendations:
        st.write(rec)
