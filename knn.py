import pandas as pd
import numpy as np
from operator import itemgetter

# importing preprocessed data from csvs
filtered_movies_df = pd.read_csv("Data_Files/movies_filtered.csv")
genre_bins = pd.read_csv("Data_Files/genre_binaries.csv")
cast_bins = pd.read_csv("Data_Files/cast_binaries.csv")
keyword_bins = pd.read_csv("Data_Files/keyword_binaries.csv")

# takes two lists of the same length that hold binary values
def getCosineSimilarity(v1, v2):
    dot_prod = np.dot(v1, v2)
    v1_mag = np.sqrt(np.sum(v1))
    v2_mag = np.sqrt(np.sum(v2))

    # if either of the magnitudes of v1 or v2 are 0, return 0 to avoid divide by 0 error
    return dot_prod / (v1_mag * v2_mag) if v1_mag * v2_mag != 0 else 0


# find the KNN of user_movie using cosine similarity
def getKNNMovies(user_movie, k=10):

    # holds tuples in the following format -> (movie_title, movie_rating, similarity)
    similarities = [
        (
            movie,
            rating,
            # aggregating the cosine similarities of the 3 attributes
            (
                getCosineSimilarity(
                    genre_bins[user_movie].to_numpy(), genre_bins[movie].to_numpy()
                )
                + getCosineSimilarity(
                    cast_bins[user_movie].to_numpy(), cast_bins[movie].to_numpy()
                )
                + getCosineSimilarity(
                    keyword_bins[user_movie].to_numpy(), keyword_bins[movie].to_numpy()
                )
            )
            / 3,
        )
        # iterates through filtered_movies_df
        for movie, rating in zip(
            filtered_movies_df["original_title"], filtered_movies_df["vote_average"]
        )
    ]

    # gets the K movies that had the highest similarity scores (skips the first one because it will always be the user defined movie)
    similarities = sorted(similarities, key=itemgetter(2), reverse=True)[1 : k + 1]

    return similarities


# uses weighted KNN algorithm to predict the score of the movie
def weightedKNNPrediction(scores, similarities):
    # movies with higher similarities should have more weight
    weights = [1 / np.square(1 - similarity) for similarity in similarities]

    weighted_scores = np.dot(scores, weights)

    return weighted_scores / np.sum(weights)


# evaluates KNN algorithm using RMSE
def getRMSE():
    # gets the KNN for every movie in filtered_movies_df
    all_movie_nearest_neighbors = [
        getKNNMovies(movie) for movie in filtered_movies_df["original_title"]
    ]
    # gets the predicted scores for every movie in filtered_movies_df
    predicted_scores = np.array(
        [
            weightedKNNPrediction(
                [neighbor[1] for neighbor in neighbors],
                [neighbor[1] for neighbor in neighbors],
            )
            for neighbors in all_movie_nearest_neighbors
        ]
    )
    # gets the actual scores for every movie in filtered_movies_df
    actual_scores = np.array([score for score in filtered_movies_df["vote_average"]])

    return np.sqrt((np.square(predicted_scores - actual_scores)).mean())


user_movie = input("Enter a movie: ")
similarities = getKNNMovies(user_movie)

predicted_score = weightedKNNPrediction(
    [similarity[1] for similarity in similarities],
    [similarity[2] for similarity in similarities],
)

print(f"Predicted Score: {predicted_score}")
print("Top 10 Most Similar Movies Predicted Score was Based On")
for similarity in similarities:
    print(f"{similarity[0]} {similarity[1]}")
