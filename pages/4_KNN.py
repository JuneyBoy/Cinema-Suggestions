import streamlit as st
import pandas as pd
import knn


st.markdown(
    "### Enter a movie and Cinema Suggestions will show you 10 of the most similar movies based on genres, cast, and keywords as well as a predicted rating of the movie using the K Nearest Neighbors Algorithm"
)

# only needs to do this one time, so adding the cache statement will make it so this function only runs the first time the page is loaded but doesn't run again as long as the user doesn't close the tab in their browser
@st.cache
def load_data():
    # importing preprocessed data from csvs
    filtered_movies_df = pd.read_csv("Data_Files/movies_filtered.csv")
    genre_bins = pd.read_csv("Data_Files/genre_binaries.csv")
    cast_bins = pd.read_csv("Data_Files/cast_binaries.csv")
    keyword_bins = pd.read_csv("Data_Files/keyword_binaries.csv")

    return (filtered_movies_df, genre_bins, cast_bins, keyword_bins)


movie_info, genres, casts, keywords = load_data()

# get movie from user via dropdown menu that has all the movies from the preprocessed set
user_movie = st.selectbox(
    "Select a movie:", [title for title in movie_info["original_title"]]
)

# get the actual ratings and similarities for each movie
similarities = knn.getKNNMovies(user_movie, movie_info, genres, casts, keywords)
# get predicted score using Weighted KNN regression
predicted_score = knn.weightedKNNPrediction(
    [movie[1] for movie in similarities], [movie[2] for movie in similarities]
)

st.markdown("##### Predicted Score")
st.write(round(predicted_score, 2))

st.markdown("##### Actual Score")
st.write(
    round(
        movie_info.loc[movie_info["original_title"] == user_movie, "vote_average"].iloc[
            0
        ],
        2,
    )
)

# convert results into dataframe (might change this so the function returns a dataframe so no conversion is necessary)
most_similar_movies_df = pd.DataFrame(
    {
        "Movie Title": [movie[0] for movie in similarities],
        "Rating": [movie[1] for movie in similarities],
        "Similarity": [movie[2] for movie in similarities],
    }
)
# by default index starts at 0 which would look a little funny to show the user the 0th movie
most_similar_movies_df.index += 1
st.markdown("##### 10 Most Similar Movies")
# need to add formatter to round ratings
st.dataframe(most_similar_movies_df.style.format({"Rating": "{:.2f}"}))
st.markdown(
    "A similarity of 1 would mean the movie has the exact same characteristics as your chosen movie and a similarity of 0 would mean the movie has none of the same characteristics as your chosen movie"
)
