import pandas as pd
import numpy as np
from operator import itemgetter

# takes two lists of the same length that hold binary values
def getCosineSimilarity(v1, v2):
    dot_prod = np.dot(v1, v2)
    v1_mag = np.sqrt(np.sum(v1))
    v2_mag = np.sqrt(np.sum(v2))

    # if either of the magnitudes of v1 or v2 are 0, return 0 to avoid divide by 0 error
    return dot_prod / (v1_mag * v2_mag) if v1_mag * v2_mag != 0 else 0


# find the KNN of user_movie using cosine similarity
def getKNNMovies(
    user_movie,
    movies,
    ratings,
    genres,
    casts,
    keywords,
    k=5,
    get_other_user_ratings=True,
):

    # holds tuples in the following format -> (movie_title, movie_rating, similarity)
    similarities = [
        (
            movie,
            rating,
            # aggregating the cosine similarities of the 3 attributes
            (
                getCosineSimilarity(
                    genres[user_movie].to_numpy(), genres[movie].to_numpy()
                )
                + getCosineSimilarity(
                    casts[user_movie].to_numpy(), casts[movie].to_numpy()
                )
                + getCosineSimilarity(
                    keywords[user_movie].to_numpy(), keywords[movie].to_numpy()
                )
            )
            / 3,
        )
        # iterates through movies
        for movie, rating in zip(movies["original_title"], movies["vote_average"])
    ]

    if get_other_user_ratings:
        # gets the K movies that had the highest similarity scores (skips the first one because it will always be the user defined movie)
        similarities = sorted(similarities, key=itemgetter(2), reverse=True)[1:]

        recs = []
        user_avg_ratings = []
        i = 0

        # only keeps movies that have at least one verified user review in ratings_filtered.csv
        while len(recs) < k and i < len(similarities):
            user_avg_rating = avg_rating_of_rec_by_users_who_liked_chosen_movie(
                ratings, user_movie, similarities[i][0]
            )
            # skips recommendation if there are no user ratings for it
            if np.isnan(user_avg_rating):
                i += 1
                continue
            else:
                recs.append(similarities[i])
                user_avg_ratings.append(user_avg_rating)
                i += 1

        return recs, user_avg_ratings
    else:
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
def getRMSE(movies, ratings, genres, casts, keywords):
    # gets the KNN for every movie in movies
    all_movie_nearest_neighbors = [
        getKNNMovies(
            movie,
            movies,
            ratings,
            genres,
            casts,
            keywords,
            get_other_user_ratings=False,
        )
        for movie in movies["original_title"]
    ]
    # gets the predicted scores for every movie in movies
    predicted_scores = np.array(
        [
            weightedKNNPrediction(
                [neighbor[1] for neighbor in neighbors],
                [neighbor[1] for neighbor in neighbors],
            )
            for neighbors in all_movie_nearest_neighbors
        ]
    )
    # gets the actual scores for every movie in movies
    actual_scores = np.array([score for score in movies["vote_average"]])

    return np.sqrt((np.square(predicted_scores - actual_scores)).mean())


def avg_rating_of_rec_by_users_who_liked_chosen_movie(
    ratings, chosen_movie, recommendation
):
    # for users who liked the chosen movie, find the ratings the users gave to the recommendation
    ratings_for_recommendation = [
        # ratings in ratings_filtered.csv are out of 5 instead of 10
        rec_rating * 2
        for rec_rating, chosen_movie_rating in zip(
            ratings[recommendation], ratings[chosen_movie]
        )
        # filters out users who did not like chosen movie and did not review the recommendation
        if chosen_movie_rating > 2.5 and rec_rating > 0
    ]

    return np.average(ratings_for_recommendation)


filtered_movies_df = pd.read_csv("Data_Files/movies_filtered.csv")
filtered_ratings_df = pd.read_csv("Data_Files/ratings_filtered.csv")
genre_bins = pd.read_csv("Data_Files/genre_binaries.csv")
cast_bins = pd.read_csv("Data_Files/cast_binaries.csv")
keyword_bins = pd.read_csv("Data_Files/keyword_binaries.csv")

print(
    getRMSE(
        filtered_movies_df, filtered_ratings_df, genre_bins, cast_bins, keyword_bins
    )
)
