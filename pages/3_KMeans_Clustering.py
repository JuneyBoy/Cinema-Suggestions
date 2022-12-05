import pandas as pd
import streamlit as st

st.markdown(
    "### Enter a movie and Cinema Suggestions will show you 10 of the most similar movies based on genres, cast, and keywords as well as a predicted rating of the movie using the K Means Algorithm"
)

# only needs to do this one time, so adding the cache statement will make it so this function only runs the first time the page is loaded but doesn't run again as long as the user doesn't close the tab in their browser
@st.cache
def load_data():
    # importing preprocessed data from csvs
    movies_filtered_df = pd.read_csv("Data_Files/movies_filtered.csv")
    ratings_filtered_df = pd.read_csv(
        "Data_Files/ratings_filtered.csv", index_col="userId"
    )
    return (movies_filtered_df, ratings_filtered_df)


movies, ratings = load_data()

# split page into columns
c1, c2 = st.columns((1, 1))

# get list of movie titles in the dataframe
movie_title_list = [title for title in movies["original_title"]]

# select box for 3 movies
user_movie1 = c1.selectbox("Movie 1:", movie_title_list)
user_movie2 = c1.selectbox("Movie 2:", movie_title_list)
user_movie3 = c1.selectbox("Movie 3:", movie_title_list)

# rating sliders for 3 movies
user_rating1 = c2.slider(
    "Your rating for movie 1 (`%s`):" % user_movie1, 0.0, 10.0, 5.0, 0.1, key="rating1"
)
user_rating2 = c2.slider(
    "Your rating for movie 2 (`%s`):" % user_movie2, 0.0, 10.0, 5.0, 0.1, key="rating2"
)
user_rating3 = c2.slider(
    "Your rating for movie 3 (`%s`):" % user_movie3, 0.0, 10.0, 5.0, 0.1, key="rating3"
)

# button to start movie recommendation with KMeans
start_kmeans = st.button("FIND RECOMMENDATIONS")

if start_kmeans:
    # store input as dict, key: movie_title, value: rating
    user_movie_rating_dict = {
        user_movie1: user_rating1,
        user_movie2: user_rating2,
        user_movie3: user_rating3,
    }
    st.markdown(user_movie_rating_dict)
