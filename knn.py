import pandas as pd
import numpy as np
from operator import itemgetter

filtered_movies_df = pd.read_csv("Data_Files/movies_filtered.csv")
genre_bins = pd.read_csv("Data_Files/genre_binaries.csv")
cast_bins = pd.read_csv("Data_Files/cast_binaries.csv")
keyword_bins = pd.read_csv("Data_Files/keyword_binaries.csv")


def getCosineSimilarity(v1, v2):
    dot_prod = np.dot(v1, v2)
    v1_mag = np.sqrt(np.sum(v1))
    v2_mag = np.sqrt(np.sum(v2))

    # some magnitudes are 0/ causing divide by 0 errors
    return dot_prod / (v1_mag * v2_mag)


def getMovieSimilarities(user_movie, k=10):

    similarities = [
        (
            movie,
            getCosineSimilarity(
                genre_bins[user_movie].to_numpy(), genre_bins[movie].to_numpy()
            )
            + getCosineSimilarity(
                cast_bins[user_movie].to_numpy(), cast_bins[movie].to_numpy()
            )
            + getCosineSimilarity(
                keyword_bins[user_movie].to_numpy(), keyword_bins[movie].to_numpy()
            ),
        )
        for movie in filtered_movies_df["original_title"]
    ]

    similarities = sorted(similarities, key=itemgetter(1), reverse=True)[1 : k + 1]

    return similarities


def weightedKNN(similarities):
    weights = [1 / np.square(score) for score in similarities]

    weighted_scores = np.dot(similarities, weights)

    return weighted_scores / np.sum(weights)


user_movie = input("Enter a movie: ")
similarities = getMovieSimilarities(user_movie)

predicted_score = weightedKNN([similarity[1] for similarity in similarities])

print(f"Predicted Score: {predicted_score}")
print("Top 10 Most Similar Movies Predicted Score was Based On")
for similarity in similarities:
    print(f"{similarity[0]}: {similarity[1]}")
