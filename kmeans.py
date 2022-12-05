'''
Dataset from: https://www.kaggle.com/datasets/rounakbanik/the-movies-dataset?resource=download&select=movies_metadata.csv
Reference: https://medium.com/nerd-for-tech/k-means-python-implementation-from-scratch-8400f30b8e5c
Reference: https://asdkazmi.medium.com/ai-movies-recommendation-system-with-clustering-based-k-means-algorithm-f04467e02fcd
'''

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
import seaborn as sns
from scipy.sparse import csr_matrix
from mpl_toolkits.axes_grid1 import make_axes_locatable
from sklearn.cluster import KMeans
from sklearn.metrics import mean_squared_error
import itertools
from sklearn.metrics import silhouette_samples, silhouette_score
import warnings
warnings.filterwarnings('ignore')

#Defining centroid of the data
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
def calculate_error(a,b):
    '''
    Given two Numpy Arrays, calculates the root of the sum of squared errores.
    '''
    error = np.square(np.sum((a-b)**2))

    return error    
'''
def sum_of_squared_errors():
    errors = np.array([])
    for centroid in range(centroids.shape[0]):
        error = calculate_error(centroids.iloc[centroid, :2], data.iloc[0,:2])
        errors = np.append(errors, error)
'''
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
    centroids = data.sample(n=k).values
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


def load_data():
    # importing preprocessed data from csvs
    filtered_movies_df = pd.read_csv("Data_Files/movies.csv", usecols = ['id', 'genres', 'original_title'])
    filtered_ratings = pd.read_csv("Data_Files/ratings.csv", usecols = ['userId','movieId','rating'])
    return (
        filtered_movies_df,
        filtered_ratings
    )

#Data Pre-Processing
def preprocess(data):
    #data = data.dropna()
    X = data.iloc[:,:-1].values
    le = preprocessing.LabelEncoder()
    
    for i in range(len(data)):
        try:
            data.reset_index(level=0, inplace=True)
            X[:,i] = le.fit_transform(X[:,i])
        except:
            break

    scaler = StandardScaler()
    for i in range(len(data)):
        try:
            if X[:,i].isnumeric():
                data = scaler.fit_transform(X[:,i])
        except:
            continue
    
movies, ratings = load_data()
preprocess(movies)
preprocess(ratings)
#data = ratings.drop_duplicates(subset="movieId", keep="first")    
ratings['movieId'] = ratings.movieId.astype(int)
movies['id'] = movies.id.astype(int)

#Creating a merged dataset
data = pd.merge(movies, ratings, left_on= ['id'], right_on= ['movieId'], how = 'left')
data = data.dropna()
data = data.reset_index(drop=True)
data.drop(data.columns[[0,1,2,3,5,6]], axis=1, inplace=True)
data.drop_duplicates(inplace=True)
print(data.columns)
#print(data)
#data = data.drop(labels=0, axis=0)
#data.columns = data.loc[0]
#data = data.drop(0)
print(data)

rate_data = data.loc[:,['movieId', 'rating']]
rate_data.columns = rate_data.loc[0]
rate_data = rate_data.drop(0)
#rate_data[]
#rate_data.columns.values[0] = pd.to_numeric(rate_data.columns[0])
#data.columns.values[1] = pd.to_numeric(data.columns[1])
#data.columns.values[2] = pd.to_numeric(data.columns[2])
#data.columns.values[3] = pd.to_numeric(data.columns[3])
#print(type(data.columns[0]))
#print(type(data.columns[1]))
#print(type(data.columns[2]))
#print(type(data.columns[3]))

#Creating wcss plot for best K
wcss(rate_data.values)
