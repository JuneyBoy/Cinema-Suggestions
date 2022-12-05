'''
Dataset from: https://www.kaggle.com/datasets/rounakbanik/the-movies-dataset?resource=download&select=movies_metadata.csv
Reference: https://medium.com/nerd-for-tech/k-means-python-implementation-from-scratch-8400f30b8e5c
Reference: https://asdkazmi.medium.com/ai-movies-recommendation-system-with-clustering-based-k-means-algorithm-f04467e02fcd
'''

import itertools
import warnings
warnings.filterwarnings('ignore')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.sparse import csr_matrix
from sklearn import preprocessing
from sklearn.cluster import KMeans
from sklearn.metrics import (mean_squared_error, silhouette_samples,
                             silhouette_score)
from sklearn.preprocessing import StandardScaler
import knn


def generate_clusters_for_ratings():
    filtered_ratings = pd.read_csv("Data_Files/ratings_filtered.csv")
    sparse_ratings = csr_matrix(filtered_ratings.drop('userId', axis=1).values)
    kmeans_obj = kmeans(sparse_ratings, 4)
    filtered_ratings['cluster'] = kmeans_obj.cluster
    filtered_ratings.to_csv("Data_Files/clustered_ratings.csv")
    #Creating wcss plot for best K
    wcss(filtered_ratings.values)

#Where k = 4 based on wcss plot done on dataset
#and X = data.values
def kmeans(X, k):
    '''
    Given a dataset and number of clusters, it clusterizes the data. 
    data: a DataFrame with all information necessary
    k: number of clusters to create
    '''

    diff = 1
    cluster = np.zeros(X.shape[0])
    centroids = X
    while diff:
        # for each observation
        for i, row in enumerate(X):
            mn_dist = float('inf')
            # dist of the point from all centroids
            for idx, centroid in enumerate(centroids):
                d = np.sqrt((centroid[0]-row[0])**2 + (centroid[1]-row[1])**2)
                # store closest centroid
                if mn_dist > d:
                    mn_dist = d
                    cluster[i] = idx
        new_centroids = pd.DataFrame(X).groupby(by=cluster).mean().values
        # if centroids are same then leave
        if np.count_nonzero(centroids-new_centroids) == 0:
            diff = 0
        else:
            centroids = new_centroids
    return centroids, cluster

    # Initialize centroids
    '''centroids = initialize_centroids(k, data)
    error = []
    compr = True
    i = 0

    while(compr):
        # Obtain centroids and error
        data['centroid'], iter_error = assign_centroid(data,centroids)
        error = np.append(error, sum(iter_error))
        # Recalculate centroids
        centroids = data.groupby('centroid').agg('mean').reset_index(drop = True)

        # Re initialize centroids
        if(centroids.shape[0] < k):
            warnings.warn("Cluster devanished! Consider reducing the number of k")
            #raise Warning("Vanished centroid. Consider changing the number of clusters.")
            number_centroids_reinitialize = k - centroids.shape[0] 
            reinitialized_centroids = initialize_centroids(number_centroids_reinitialize, data.drop(['centroid'], axis = 1))

            # Find the index of the centroids that  are missing
            ind_missing = np.isin(np.array(range(k)), centroids.index)
            reinitialized_centroids.index = np.array(range(k))[ind_missing == False]

            # Include the new centroids
            centroids = centroids.append(reinitialized_centroids)

        # Check if the error has decreased
        if(len(error)<2):
            compr = True
        else:
            if(round(error[i],3) !=  round(error[i-1],3)):
                compr = True
            else:
                compr = False
        i = i + 1 


    #data['centroid'], iter_error = assign_centroid(data,centroids)
    #centroids = data.groupby('centroid').agg('mean').reset_index(drop = True)

    return (data['centroid'], error[-1], centroids)'''

#Calculating within clusters sum of square (WSSC)
def calculate_cost(X, centroids, cluster):
  sum = 0
  for i, val in enumerate(X):
    sum += np.sqrt((centroids[int(cluster[i]), 0]-val[0])**2 +(centroids[int(cluster[i]), 1]-val[1])**2)
  return sum

def get_cluster_for_user(movies):
    similarity = {}
    max_center = (-1,-1)
    #Initialize array that is the same length
    ratings_arr = [0] * (len(ratings.columns)-2)
    for movie, rating in movies.items():
        ratings_arr [ratings.columns.get_loc(movie)] = rating 
    for center, label in zip(centers, clusternum):
        sim = knn.getCosineSimilarity(ratings_arr, center)
        similarity[label] = sim
        curr_center = center
        if curr_center > max_center[1]:
            max_center = (label, curr_center) 
    
    return(max_center[0])

# *********** TODO: Has some K MEANS package ideologies    ********************
def get_top_movies_from_cluster(kmeans, all_movies, all_ratings, movie_ratings, num_of_movies_to_show=5):
    # assign cluster to user
    cluster_num = kmeans.predict([movie_ratings])
    print(cluster_num)
    # filter ratings to get the all the users in the same cluster as our user
    users_in_cluster = all_ratings.loc[all_ratings['cluster'] == cluster_num[0]]
    print(users_in_cluster)
    users_in_cluster = users_in_cluster.replace(0, np.nan)
    # get their avgs for every movie (excluse userId and cluster columns)
    movie_avgs = {movie : np.nanmean(users_in_cluster[movie]) for movie in users_in_cluster.columns[1:-1]}
    # keep the movies with the top avgs

    #sorted_movie_avgs = sorted(movie_avgs.items(), key=lambda item: item[1], reverse=True)

    sorted_movie_avgs = sorted([(movie, avg) for movie, avg in movie_avgs.items() if not np.isnan(avg)], key=lambda item: item[1], reverse=True)[:num_of_movies_to_show]

    # return list of tuple where the tuple is (movie_title, rating average from all users, rating average of users in same cluster)
    return [(movie[0], all_movies[all_movies['original_title']==movie[0]]['vote_average'].iloc[0], movie[1] * 2) for movie in sorted_movie_avgs]

def calculate_error(a,b):
    '''
    Given two Numpy Arrays, calculates the root of the sum of squared errores.
    '''
    error = np.square(np.sum((a-b)**2))

    return error    

def wcss(X):
    cost_list = []
    for k in range(1, 10):
        centroids, cluster = kmeans(X, k)
        # WCSS (Within cluster sum of square)
        cost = calculate_cost(X, centroids, cluster)
        cost_list.append(cost)

    #Plot line between WSSC and k
    #You will know what k should be based on where it reduces less
    sns.lineplot(x=range(1,10), y=cost_list, marker='o')
    plt.xlabel('k')
    plt.ylabel('WCSS')
    plt.show()

 ################################################# STUFF THAT MAY OR MAY NOT BE NEEDED ###############################################   

#Defining centroid of the data
# MAY OR MAY NOT BE NEEDED HERE FOR POSSIBILITY
'''
def initialize_centroids(k, data):

    n_dims = data.shape[1]
    centroid_min = data.min().min()
    centroid_max = data.max().max()
    centroids = []

    for centroid in range(k):
        centroid = np.random.uniform(centroid_min, centroid_max, n_dims)
        centroids.append(centroid)

    centroids = pd.DataFrame(centroids, columns = data.columns)

    return centroids
'''

# MAY OR MAY NOT BE NEEDED HERE FOR POSSIBILITY    
'''
def sum_of_squared_errors():
    errors = np.array([])
    for centroid in range(centroids.shape[0]):
        error = calculate_error(centroids.iloc[centroid, :2], data.iloc[0,:2])
        errors = np.append(errors, error)
'''

# MAY OR MAY NOT BE NEEDED HERE FOR POSSIBILITY
def assign_centroid(data, centroids):
    '''
    Receives a dataframe of data and centroids and returns a list assigning each observation a centroid.
    data: a dataframe with all data that will be used.
    centroids: a dataframe with the centroids. For assignment the index will be used.
    '''

    n_observations = data.shape[0]
    centroid_assign = []
    centroid_errors = []
    k = centroids.shape[0]


    for observation in range(n_observations):

        # Calculate the errror
        errors = np.array([])
        for centroid in range(k):
            error = calculate_error(centroids.iloc[centroid, :2], data.iloc[observation,:2])
            errors = np.append(errors, error)

        # Calculate closest centroid & error 
        closest_centroid =  np.where(errors == np.amin(errors))[0].tolist()[0]
        centroid_error = np.amin(errors)

        # Assign values to lists
        centroid_assign.append(closest_centroid)
        centroid_errors.append(centroid_error)

    return (centroid_assign,centroid_errors)


