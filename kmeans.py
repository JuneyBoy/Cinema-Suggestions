import warnings

warnings.filterwarnings("ignore")
# import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def init_centroids(k, ratings):
    # each dimension of the centroid will be a movie
    # subtract 1 since the first column is user IDs
    num_of_dimensions = len(ratings.columns) - 1

    # intialize centroids randomly
    centroids = [
        # min 0f 0, max of 5 with heavy bias dimension for 0 (since vast majority of ratings are 0)
        np.random.choice(6, num_of_dimensions, p=[0.8, 0.025, 0.05, 0.05, 0.05, 0.025])
        for _ in range(k)
    ]

    centroid_df = pd.DataFrame(centroids, columns=ratings.columns[1:])

    centroid_df.index.names = ["Cluster_Label"]

    return centroid_df


# Calculating within clusters sum of square (WSSC)
def calculate_error(data_point, centroid):
    return np.linalg.norm(data_point - centroid)


# find which cluster each user from the data is in, and return
def assign_clusters_to_users(ratings, centroids):
    cluster_labels = []
    errors = []

    for _, row in ratings.iterrows():
        label, error = get_cluster_for_user(row, centroids)
        cluster_labels.append(label)
        errors.append(error)

    return cluster_labels, np.sum(errors)


# assign user to a cluster based on which centroid they are closest too
def get_cluster_for_user(movie_ratings, centroids):
    # format of tuple is (centroid_label, error)
    closest_centroid = (float("inf"), float("inf"))
    # Initialize array that is the same length
    for index, row in centroids.iterrows():
        error = calculate_error(movie_ratings, row)
        # if the error from the current cluster is smaller than the previous smallest error, reassign it
        if error < closest_centroid[1]:
            closest_centroid = (index, error)
    # return cluster label and
    return closest_centroid[0], error


# run this once to identify best clusters and save csv file with instances labeled
def k_means(k, ratings):
    num_of_iterations = 0
    previous_sse = float("inf")
    current_sse = 0
    # limit iterations to 5
    while num_of_iterations < 5:
        # assign centroids randomly
        centroids = init_centroids(k, ratings)
        # get the labels and SSE for this clustering
        labels, current_sse = assign_clusters_to_users(ratings, centroids)
        # if the current SSE is less than the previous iteration, then break
        if current_sse > previous_sse:
            ratings["cluster"] = labels
            ratings.to_csv("Data_Files/ratings_clustered.csv", index=False)
            centroids.to_csv("Data_Files/centroids.csv")
            return previous_sse
        num_of_iterations += 1

    return current_sse


def get_top_movies_from_cluster(
    user_ratings, all_movies, all_ratings, centroids, num_of_movies_to_show=5
):
    # assign cluster to user
    cluster_num, error = get_cluster_for_user(user_ratings, centroids[1:])
    # filter ratings to get the all the users in the same cluster as our user
    users_in_cluster = all_ratings.loc[all_ratings["cluster"] == cluster_num]
    users_in_cluster.drop("cluster", axis=1, inplace=True)
    # replace 0s with NaN so 0s do not skew movie rating averages
    users_in_cluster.replace(0, np.nan, inplace=True)
    # get their avgs for every movie (exclude userId and cluster columns)
    movie_avgs = {
        movie: np.nanmean(users_in_cluster[movie])
        for movie in users_in_cluster.columns[1:-1]
    }

    # keep the movies with the top avgs
    sorted_movie_avgs = sorted(
        [(movie, avg) for movie, avg in movie_avgs.items() if not np.isnan(avg)],
        key=lambda item: item[1],
        reverse=True,
    )[:num_of_movies_to_show]

    # return list of tuple where the tuple is (movie_title, rating average from all users, rating average of users in same cluster)
    return (
        [
            (
                movie[0],
                all_movies[all_movies["original_title"] == movie[0]][
                    "vote_average"
                ].iloc[0],
                movie[1] * 2,
            )
            for movie in sorted_movie_avgs
        ],
        error,
    )


# k_means(4, pd.read_csv("Data_Files/ratings_filtered.csv"))

# call kmeans 20 times, and plot the SSE for each run to find optimal value of k
def generate_elbow_plot(ks_to_test=10):
    errors = [
        k_means(k, pd.read_csv("Data_Files/ratings_filtered.csv"))
        for k in range(1, ks_to_test)
    ]

    plt.plot([k for k in range(1, ks_to_test)], errors)
    plt.title("Elbow Plot")
    plt.xlabel("k")
    plt.ylabel("SSE")

    plt.show()


# generate_elbow_plot()
