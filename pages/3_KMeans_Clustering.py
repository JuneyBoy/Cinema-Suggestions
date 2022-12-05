import streamlit as st
from sklearn.preprocessing import StandardScaler
import pandas as pd
import kmeans


st.markdown(
    "### Enter a movie and Cinema Suggestions will show you 10 of the most similar movies based on genres, cast, and keywords as well as a predicted rating of the movie using the K Means Algorithm"
)

# only needs to do this one time, so adding the cache statement will make it so this function only runs the first time the page is loaded but doesn't run again as long as the user doesn't close the tab in their browser
@st.cache
def load_data():
    # importing preprocessed data from csvs
    filtered_movies_df = pd.read_csv("Data_Files/movies.csv", usecols = ['id', 'genres', 'original_title'])
    filtered_ratings = pd.read_csv("Data_Files/ratings.csv", usecols = ['movieId','rating'])
    return (
        filtered_movies_df,
        filtered_ratings
    )

movies, ratings = load_data()

#Creating a merged dataset
data = pd.merge(movies, ratings, left_on= ['id'], right_on= ['movieId'], how = 'left')

#Data Pre-Processing
data.dropna(inplace= True )
data.drop_duplicates(subset="id", keep="first")
scaler = StandardScaler()
data = scaler.fit_transform(data)

#Creating wcss plot for best K
kmeans.wcss(data)

# get movie from user via dropdown menu that has all the movies from the preprocessed set
user_movie = st.selectbox(
    "Select a movie:", [title for title in movies["original_title"]]
)
'''
k_neighbors = st.slider(
    "How many recommendations do you want (will define K in KNN algorithm)",
    min_value=1,
    max_value=10,
)'''

if st.button("FIND RECOMMENDATIONS"):
    # get the actual ratings and similarities for each movie
    recommendations, ratings_of_recs_from_other_users = knn.getKNNMovies(
        user_movie, movies, ratings, genres, casts, keywords, k_neighbors
    )
    # get predicted score using Weighted KNN regression
    predicted_score_all_ratings = knn.weightedKNNPrediction(
        [movie[1] for movie in recommendations], [movie[2] for movie in recommendations]
    )
    # get predicted score using Weighted KNN regression
    predicted_score_users_who_liked_chosen_movie = knn.weightedKNNPrediction(
        ratings_of_recs_from_other_users, [movie[2] for movie in recommendations]
    )

    actual_score = movies.loc[
        movies["original_title"] == user_movie, "vote_average"
    ].iloc[0]

    st.write("Actual Score: ", round(actual_score, 2))

    st.write(
        "Predicted Score (based on all user ratings): ",
        round(predicted_score_all_ratings, 2),
    )
    st.write(
        "Error (%): ",
        round(
            abs((actual_score - predicted_score_all_ratings) / actual_score) * 100, 2
        ),
    )

    st.write(
        f"Predicted Score (based on ratings of users who liked {user_movie}): ",
        round(predicted_score_users_who_liked_chosen_movie, 2),
    )

    # convert results into dataframe (might change this so the function returns a dataframe so no conversion is necessary)
    most_similar_movies_df = pd.DataFrame(
        {
            "Movie Title": [movie[0] for movie in recommendations],
            "Avg Rating of All Users": [movie[1] for movie in recommendations],
            f"Avg Rating of Users Who Liked {user_movie}": [
                rating for rating in ratings_of_recs_from_other_users
            ],
            "Similarity": [movie[2] for movie in recommendations],
        }
    )
    # by default index starts at 0 which would look a little funny to show the user the 0th movie
    most_similar_movies_df.index += 1
    st.markdown("##### 10 Most Similar Movies")
    # need to add formatter to round ratings
    st.table(
        most_similar_movies_df.style.format(
            {
                "Avg Rating of All Users": "{:.2f}",
                f"Avg Rating of Users Who Liked {user_movie}": "{:.2f}",
            }
        )
    )
    st.markdown(
        "A similarity of 1 would mean the movie has the exact same characteristics as your chosen movie and a similarity of 0 would mean the movie has none of the same characteristics as your chosen movie."
    )
