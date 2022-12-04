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
def getKNNMovies(user_movie, movies, ratings, genres, casts, keywords, k=10):

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

    # gets the K movies that had the highest similarity scores (skips the first one because it will always be the user defined movie)
    similarities = sorted(similarities, key=itemgetter(2), reverse=True)[1:]

    recs = []
    user_avg_ratings = []
    i = 0

    # only keeps movies that have at least one verified user review in ratings_filtered.csv
    while len(recs) < k:
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
        getKNNMovies(movie, ratings, movies, genres, casts, keywords)
        for movie in zip(movies["original_title"])
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


# user_movie = input("Enter a movie: ")
# similarities = getKNNMovies(user_movie)

# predicted_score = weightedKNNPrediction(
#     [similarity[1] for similarity in similarities],
#     [similarity[2] for similarity in similarities],
# )

# print(f"Predicted Score: {predicted_score}")
# print("Top 10 Most Similar Movies Predicted Score was Based On")
# for similarity in similarities:
#     print(f"{similarity[0]} {similarity[1]}")
