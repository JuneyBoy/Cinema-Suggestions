import pandas as pd
import streamlit as st
import time
import kmeans
import numpy as np

start_time = time.time()

st.markdown(
    "### Enter a movie and Cinema Suggestions will show you 10 of the most similar movies based on clustering user ratings using the K-Means Algorithm"
)

# only needs to do this one time, so adding the cache statement will make it so this function only runs the first time the page is loaded but doesn't run
# again as long as the user doesn't close the tab in their browser
@st.cache
def load_data():
    # importing preprocessed data from csvs
    filtered_movies = pd.read_csv(
        "Data_Files/movies_filtered.csv",
        usecols=["id", "genres", "original_title", "vote_average"],
    )
    filtered_ratings = pd.read_csv("Data_Files/ratings_clustered.csv")
    centroids = pd.read_csv("Data_Files/centroids.csv")

    filtered_ratings.set_index(["userId"], inplace=True)
    centroids.set_index(["Cluster_Label"], inplace=True)

    return (filtered_movies, filtered_ratings, centroids)


movies, ratings, centroids = load_data()

# split page into columns
c1, c2 = st.columns((1, 1))

# get list of movie titles in the dataframe
movie_title_list = [title for title in movies["original_title"]]

# select box for 3 movies
user_movie1 = c1.selectbox("Movie 1:", movie_title_list, index=0)
user_movie2 = c1.selectbox("Movie 2:", movie_title_list, index=1)
user_movie3 = c1.selectbox("Movie 3:", movie_title_list, index=2)

# rating sliders for 3 movies
user_rating1 = c2.slider(
    "Your rating for Movie 1 (`%s`):" % user_movie1, 0.0, 10.0, 5.0, 0.5, key="rating1"
)
user_rating2 = c2.slider(
    "Your rating for Movie 2 (`%s`):" % user_movie2, 0.0, 10.0, 5.0, 0.5, key="rating2"
)
user_rating3 = c2.slider(
    "Your rating for Movie 3 (`%s`):" % user_movie3, 0.0, 10.0, 5.0, 0.5, key="rating3"
)

# button to start movie recommendation with KMeans
start_kmeans = st.button("FIND RECOMMENDATIONS")

if start_kmeans:
    # store input as dict, key: movie_title, value: rating
    # dividing by 2 because the rating is out of 10 where 10 = 5
    user_movie_rating_dict = {
        user_movie1: user_rating1 / 2,
        user_movie2: user_rating2 / 2,
        user_movie3: user_rating3 / 2,
    }

    # initialize ratings for user
    ratings_arr = [0] * (len(ratings.columns) - 1)
    # update the values of the array based on the ratings the user entered frmo the GUI
    for movie, rating in user_movie_rating_dict.items():
        ratings_arr[ratings.columns.get_loc(movie)] = rating

    top_movies, error = kmeans.get_top_movies_from_cluster(
        ratings_arr, movies, ratings, centroids
    )

    most_similar_movies_df = pd.DataFrame(
        {
            "Movie Title": [movie[0] for movie in top_movies],
            "Avg Rating of All Users": [movie[1] for movie in top_movies],
            "Avg Rating of Users Similar to You": [movie[2] for movie in top_movies],
        }
    )

    st.table(
        most_similar_movies_df.style.format(
            {
                "Avg Rating of All Users": "{:.2f}",
                f"Avg Rating of Users Similar to You": "{:.2f}",
            }
        )
    )

    st.write("Time taken to get recommendations (seconds): ", time.time() - start_time)
    st.write(
        "Mean of Avg Ratings From Similar Users: ",
        np.average(most_similar_movies_df[f"Avg Rating of Users Similar to You"]),
    )
    st.write("Distance Between User and Cluster Center: ", error)
