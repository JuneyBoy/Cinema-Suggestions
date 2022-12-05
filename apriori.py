from itertools import combinations

import numpy as np
import pandas as pd


def load_data():
    # load the data from the csv files
    movies_df = pd.read_csv("Data_Files/movies_filtered.csv")
    ratings_df = pd.read_csv("Data_Files/ratings_small.csv")
    ratings_filtered_df = pd.read_csv("Data_Files/ratings_filtered.csv")
    df = pd.read_csv("Data_Files/ratings_filtered.csv", index_col="userId")
    return (movies_df, ratings_df, ratings_filtered_df, df)


def encode_ratings(df, rating_threshold=3):
    """
    Returns a DataFrame with rating values converted to True or False, based on rating threshold.

    Parameters
    -----------
    df: pandas DataFrame
      The DataFrame with:
      - userId as the id of each row,
      - movie titles as column names,
      - the user's rating of that movie (0.0-5.0) as the value.

    rating_threshold: numeric (default: 3)
      The threshold for which movie ratings above the threshold will be encoded to 1,
      and movie ratings below the threshold will be encoded to 0.

    Returns
    -----------
    pandas DataFrame with:
    - userId as the id of each row,
    - movie titles as column names,
    - True/False as the value if the user's rating of that movie (0.0-5.0) >= `rating_threshold`.
    """
    return df >= rating_threshold


def combinations_generator(old_combinations):
    """
    Generates all combinations based on previous state of Apriori algorithm.

    Parameters
    -----------
    old_combinations: np.array
      All combinations (represented by a matrix) with high enough support in the previous step.
      # of columns == combination size of previous step.
      Each row represents one combination and contains item indexes in ascending order.

    Returns
    -----------
    Generator of all combinations from (last step) x (items from the previous step).
    """
    previous_items = np.unique(old_combinations.flatten())
    for old_combination in old_combinations:
        # last value from old combination is max column index
        max_combination = old_combination[-1]

        # get all items from old combination that are greater than last value
        valid_items = previous_items[previous_items > max_combination]

        # convert array to tuple
        old_tuple = tuple(old_combination)

        # generate valid itemsets
        for item in valid_items:
            # use yield instead of return, which will output <generator> object
            # good for large data, good for runtime and memory
            yield from old_tuple
            yield item


def apriori(df, min_support=0.1, max_k_itemsets=3):
    """
    Get frequent itemsets from a DataFrame with
      - id for each transaction id
      - columns with names as 1-itemsets and values as True/False (True if the itemset is in the transaction)
      - rows as each transaction

    Parameters
    -----------
    df : pandas DataFrame
      DataFrame with id field and columns with values as True/False.

    min_support : float (default: 0.1)
      Minimum support threshold of the itemsets returned.
      `support = # of transactions with item(s) / total # of transactions`.

    max_k_itemsets : int (default: 3)
      Maximum size of the itemsets generated.

    Returns
    -----------
    pandas DataFrame with columns ['support', 'itemsets'] of all itemsets
      that have support >= `min_support` and itemset length < `max_k_itemsets`.
      Each itemset is of type `frozenset` (a Python built-in type), which is immutable.
    """

    def calculate_support(x, num_rows):
        """
        Calculate support as the column-wise sum of values / number of rows

        Parameters
        -----------
        x : matrix of True/False
          matrix containing only True/False values

        num_rows : numeric
          number of rows in x

        Returns
        -----------
        np.array containing the support values
        """
        # array containing sum of each column, divided by the number of rows
        out = np.sum(x, axis=0) / num_rows
        return np.array(out)

    # check that min_support is valid
    if min_support < 0.0 or min_support >= 1.0:
        raise ValueError(
            "`min_support` must be within the interval `(0, 1]`. Got %s." % min_support
        )

    # check that max_k_itemsets is valid
    if max_k_itemsets < 0:
        raise ValueError(
            "`max_k_itemsets` must be greater than 0`. Got %s." % max_k_itemsets
        )

    # start apriori with the singular values (1-itemsets)
    # extract values from data frame
    X = df.values

    # calculate the supports of each column in data frame (each itemset)
    supports = calculate_support(X, X.shape[0])

    # create np array of column indexes
    col_idx_arr = np.arange(X.shape[1])

    # dictionaries storing supports and itemsets
    # support_dict: key: k, value: support of the k-itemset
    # itemset_dict: key: k, value: k-itemsets (indexes of columns from OG df)
    support_dict = {1: supports[supports >= min_support]}
    itemset_dict = {1: col_idx_arr[supports >= min_support].reshape(-1, 1)}

    # continue apriori after 1-itemsets
    k = 1
    while k < max_k_itemsets:
        k_next = k + 1

        # CANDIDATE-GENERATION
        # get generator of new itemsets, convert to np array
        # reshape into matrix of rows=any, cols=k_next
        candidates_generator = combinations_generator(itemset_dict[k])
        candidates = np.fromiter(candidates_generator, dtype=int).reshape(-1, k_next)

        # no new combinations => stop
        if candidates.size == 0:
            # exit condition: no new combinations
            break

        # get new array of transaction data from the candidates
        candidates_df = np.all(X[:, candidates], axis=2)

        # calculate supports for new combinations
        supports = calculate_support(np.array(candidates_df), X.shape[0])

        # CANDIDATE-PRUNING
        # populate supports and itemsets only if the supports are above threshold
        support_mask = supports >= min_support
        if any(support_mask):
            itemset_dict[k_next] = np.array(candidates[support_mask])
            support_dict[k_next] = np.array(supports[support_mask])
            k = k_next
        else:
            # exit condition: no more itemsets that pass the support threshold
            break

    # add each support and itemset to list
    # keep it in pandas form, so use Series
    support_itemset_list = []
    for k in sorted(itemset_dict):
        supports = pd.Series(support_dict[k])
        itemsets = pd.Series([frozenset(i) for i in itemset_dict[k]], dtype="object")
        pair = pd.concat((supports, itemsets), axis=1)
        support_itemset_list.append(pair)

    # add the list of support-itemset pairs to a dataframe
    # change column names to: "support" and "itemsets"
    freq_itemsets_df = pd.concat(support_itemset_list)
    freq_itemsets_df.columns = ["support", "itemsets"]

    # replace all column indexes with the corresponding column name
    mapping = {idx: item for idx, item in enumerate(df.columns)}
    freq_itemsets_df["itemsets"] = freq_itemsets_df["itemsets"].apply(
        lambda x: frozenset([mapping[i] for i in x])
    )

    # reset the indexes of the data frame
    freq_itemsets_df.reset_index(drop=True, inplace=True)

    return freq_itemsets_df


def create_association_rules(frequent_itemsets, metric="support"):
    """
      Generates a DataFrame of association rules including the metrics 'support', 'confidence', and 'lift'.

      Parameters
      -----------
      frequent_itemsets : pandas DataFrame
        DF of frequent itemsets with columns ['support', 'itemsets']

      metric : string (default: 'support')
        Metric to evaluate if a rule is of interest: 'support', 'confidence', 'lift'.

      Returns
      ----------
      pandas DataFrame with columns "antecedents" and "consequents",
        and metric columns: "antecedent support", "consequent support", "support", "confidence", "lift".
        Each entry in the "antecedents" and "consequents" columns are
    of type `frozenset` (a Python built-in type), which is immutable.
    """
    # validate the frequent_itemsets DF: contains columns "support" and "itemsets"
    if not all(col in frequent_itemsets.columns for col in ["support", "itemsets"]):
        raise ValueError(
            "Dataframe must contain only the columns 'support' and 'itemsets'"
        )

    # metrics for association rules
    # sAC: total support
    # sA: antecedent support
    # sC: consequent support
    # support(A->C) = support(A+C), range: [0, 1]
    # confidence(A->C) = support(A->C) / support(A), range: [0, 1]
    # lift(A->C) = confidence(A->C) / support(C), range: [0, inf]
    metric_dict = {
        "antecedent support": lambda sAC, sA, sC: sA,
        "consequent support": lambda sAC, sA, sC: sC,
        "support": lambda sAC, sA, sC: sAC,
        "confidence": lambda sAC, sA, sC: sAC / sA,
        "lift": lambda sAC, sA, sC: metric_dict["confidence"](sAC, sA, sC) / sC,
    }
    metric_names = list(metric_dict.keys())

    # if frequent itemsets are empty, then results will be empty
    if not frequent_itemsets.shape[0]:
        # return empty df with names
        return pd.DataFrame(columns=["antecedents", "consequents"] + metric_names)

    # get dict of {frequent itemset} -> support
    itemsets = frequent_itemsets["itemsets"].values
    supports = frequent_itemsets["support"].values
    frozenset_vect_func = np.vectorize(lambda x: frozenset(x))
    frequent_items_dict = dict(zip(frozenset_vect_func(itemsets), supports))

    # prepare buckets to collect frequent rules
    rule_antecedents = []
    rule_consequents = []
    rule_supports = []

    # iterate over all frequent itemsets
    for itemset in frequent_items_dict.keys():
        # get the support value from the value of the dict
        sAC = frequent_items_dict[itemset]

        # all combinations of antecedents and consequents
        for idx in range(len(itemset) - 1, 0, -1):
            for combination in combinations(itemset, r=idx):
                antecedent = frozenset(combination)
                consequent = itemset.difference(antecedent)

                # sA: antecedent support, sC: consequent support
                sA = frequent_items_dict[antecedent]
                sC = frequent_items_dict[consequent]

                # add to metric lists
                rule_antecedents.append(antecedent)
                rule_consequents.append(consequent)
                rule_supports.append([sAC, sA, sC])

    # check if any supports were generated
    if rule_supports:
        # generate metrics
        rule_supports = np.array(rule_supports).T.astype(float)
        rules_df = pd.DataFrame(
            data=list(zip(rule_antecedents, rule_consequents)),
            columns=["antecedents", "consequents"],
        )

        # calculate the metrics given the total, antecedent, and consequent supports,
        # and add it to the dataframe in the appropriate column
        sAC = rule_supports[0]
        sA = rule_supports[1]
        sC = rule_supports[2]
        for m in metric_names:
            rules_df[m] = metric_dict[m](sAC, sA, sC)

        # sort by descending order of the metric
        rules_df = rules_df.sort_values(by=[metric], ascending=False)
        return rules_df
    else:
        # return empty df with names
        return pd.DataFrame(columns=["antecedents", "consequents"] + metric_names)


def recommend_movies_apriori(
    movie_title,
    movies_df,
    ratings_filtered_df,
    rules_df,
    max_movies=10,
):
    """
    Recommends movies given a movie title.

    Parameters
    -----------
    movie_title : string
      The title of the movie to find recommendations for.
    movies_df : pandas DataFrame
      DF containing movie data. from "movies_filtered.csv"
    ratings_filtered_df : pandas DataFrame
      DF containing pre-processed ratings data from certain movies. from ""ratings_filtered.csv"
    rules_df : pandas DataFrame
      DF containing the association rules.
    max_movies : int (default: 10)
      The maximum number of recommended movies to get.

    Returns
    -----------
    pandas DataFrame with at most `max_movies` recommendations, containing:
      - the movies' titles
      - the movies' average rating from all users
      - the movies' average rating from users that also liked the inputted `movie_title`
    """
    # get dataframe with user inputted movie as the only item in antecedent (length of 1)
    movie_mask = rules_df["antecedents"].apply(
        lambda x: len(x) == 1 and movie_title in x
    )
    user_movie_rules_df = rules_df[movie_mask]

    # get all the movies (consequents) where the rule had user_movie (antecedent)
    movies = user_movie_rules_df["consequents"].values

    # list of unique movies, appended in order of descending lift
    recommended_movies = []
    for movie in movies:
        for title in movie:
            if title not in recommended_movies:
                recommended_movies.append(title)

    # get only up to `max_movie` number of recommendations
    recommended_movies = recommended_movies[0:max_movies]
    avg_ratings = []
    user_avg_ratings = []
    for movie in recommended_movies:
        # average rating from all users
        avg_rating = np.average(
            movies_df[movies_df["original_title"] == movie]["vote_average"]
        )
        avg_ratings.append(avg_rating)

        # average rating from users who liked the recommended movie
        user_avg_rating = avg_rating_of_rec_by_users_who_liked_chosen_movie(
            ratings=ratings_filtered_df,
            chosen_movie=movie_title,
            recommendation=movie,
        )
        # skips recommendation if there are no user ratings for it
        if np.isnan(user_avg_rating):
            continue
        else:
            user_avg_ratings.append(user_avg_rating)

    # create new DF to display recommendations
    recommended_movies_df = pd.DataFrame(
        {
            "Movie Title": recommended_movies,
            "Avg Rating of All Users": avg_ratings,
            f"Avg Rating of Users Who Liked {movie_title}": user_avg_ratings,
        }
    )

    # output user movie df, and recommended movie df
    return user_movie_rules_df, recommended_movies_df


def avg_rating_of_rec_by_users_who_liked_chosen_movie(
    ratings, chosen_movie, recommendation
):
    """
    Get the ratings of a `recommendation` movie from users that liked the `chosen_movie`.
    `ratings` must be the DataFrame containing user ratings for each movie
    (userId as id, movie_title as columns, rating as values).
    """
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


"""
# constants for algorithms
RATING_THRESHOLD = 3
MIN_SUPPORT = 0.07
MAX_K_ITEMSETS = 5
METRIC = "lift"

# load data
movies_df, ratings_df, ratings_filtered_df, df = load_data()

# our apriori model needs data in a matrix with:
# userId is index, columns are movie tiles, and the values
# are False/True depending on whether user rated movie over some threshold.
df = df.applymap(encode_ratings, None, rating_threshold=RATING_THRESHOLD)

# generate the frequent itemsets using the apriori algorithm
freq_itemsets = apriori(df, min_support=MIN_SUPPORT, max_k_itemsets=MAX_K_ITEMSETS)

# support: probabilty of users watching movie M1
# support(M) = (# user watchlists containing M) / (# user watchlists)
# confidence: out of total users having watched movie M1, how many have also watched movie M2
# confidence(M1 -> M2) = (# user watchlists containing M1 and M2) / (# user watchlists containing M1)
# lift: ratio of confidence and support
# lift(M1 -> M2) = confidence(M1 -> M2) / support(M2)
# high lift suggests there is some relation between the two movies and most of the
# users who have watched movie M1 are also likely to watch movie M2.
# rules are sorted by descending value of the given metric
rules_df = create_association_rules(freq_itemsets, metric=METRIC)

# get recommended movies for the user inputted movie
while True:
    user_movie = input("Enter a movie (`Quit` to quit): ")
    if user_movie.lower() == "Quit".lower():
        break
    if user_movie not in movies_df["title"].values:
        print("Movie not found in dataset.")
    else:
        user_movie_rules_df, recommended_movies = recommend_movies_apriori(
            movie_title=user_movie,
            movies_df=movies_df,
            ratings_filtered_df=ratings_filtered_df,
            rules_df=rules_df,
            max_movies=10,
        )
        print(freq_itemsets)
        print(rules_df)
        print(user_movie_rules_df)
        print(recommended_movies)
        for movie in recommended_movies:
            print(movie)
"""
