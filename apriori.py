from itertools import combinations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def load_data():
    # load the data
    ratings_df = pd.read_csv("Data_Files/ratings_small.csv")
    movies_df = pd.read_csv("Data_Files/movies_filtered.csv")
    df = pd.read_csv("Data_Files/ratings_filtered.csv", index_col="userId")

    return (movies_df, ratings_df, df)


def plot_ratings_distribution(ratings_df):
    plt.figure(figsize=(10, 5))
    ax = sns.countplot(data=ratings_df, x="rating")
    labels = ratings_df["rating"].value_counts().sort_index()
    plt.title("Distribution of Ratings")
    plt.xlabel("Ratings")

    for i, v in enumerate(labels):
        ax.text(
            i, v + 100, str(v), horizontalalignment="center", size=14, color="black"
        )
    plt.show()


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
      All combinations (represented by a matrix) that have high enough support in the previous step.
      Number of columns is equal to the combination size of the previous step.
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


def apriori(df, min_support=0.1, max_len=None):
    """
    Get frequent itemsets from a DataFrame with
      - id for each transaction id
      - columns with names as 1-itemsets and values as True/False (True if the itemset is in the transaction)
      - rows as each transaction

    Parameters
    -----------
    df : pandas DataFrame
      DataFrame with id field and columns with values as True/False.
      For example,
    ```
        id    A       B         C       D
        0     True    False     True    False
        1     True    True      True    False
        2     True    False     True    False
    ```

    min_support : float (default: 0.1)
      Minimum support threshold of the itemsets returned.
      `support = # of transactions with item(s) / total # of transactions`.

    max_len : int (default: None)
      Maximum length of the itemsets generated. If `None` (default), any itemset lengths are evaluated.

    Returns
    -----------
    pandas DataFrame with columns ['support', 'itemsets'] of all itemsets
      that have support >= `min_support` and itemset length < `max_len`.
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
            "`min_support` must be a number within the interval `(0, 1]`. Got %s."
            % min_support
        )

    # check that max_len is valid
    if max_len != None and max_len < 0:
        raise ValueError(
            "`max_len` must be `None` or an integer greater than 0`. Got %s." % max_len
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
    while k < (max_len or float("inf")):
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
        _bools = np.all(X[:, candidates], axis=2)

        # calculate supports for new combinations
        supports = calculate_support(np.array(_bools), X.shape[0])

        # CANDIDATE-PRUNING
        # populate supports and itemsets if any of the supports are above threshold
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
    res_list = []
    for k in sorted(itemset_dict):
        supports = pd.Series(support_dict[k])
        itemsets = pd.Series([frozenset(i) for i in itemset_dict[k]], dtype="object")
        pair = pd.concat((supports, itemsets), axis=1)
        res_list.append(pair)

    # add the list of support-itemset pairs to a dataframe
    # change column names to: "support" and "itemsets"
    freq_itemsets_df = pd.concat(res_list)
    freq_itemsets_df.columns = ["support", "itemsets"]

    # replace all column indexes with the corresponding column name
    mapping = {idx: item for idx, item in enumerate(df.columns)}
    freq_itemsets_df["itemsets"] = freq_itemsets_df["itemsets"].apply(
        lambda x: frozenset([mapping[i] for i in x])
    )

    # reset the indexes of the data frame
    freq_itemsets_df.reset_index(drop=True, inplace=True)

    return freq_itemsets_df


def create_association_rules(frequent_itemsets, metric="support", metric_threshold=0.0):
    """
    Generates a DataFrame of association rules including the metrics 'support', 'confidence', and 'lift'.

    Parameters
    -----------
    frequent_itemsets : pandas DataFrame
      DF of frequent itemsets with columns ['support', 'itemsets']

    metric : string (default: 'support')
      Metric to evaluate if a rule is of interest: 'support', 'confidence', 'lift'.
      - support(A->C) = support(A+C), range: [0, 1]
      - confidence(A->C) = support(A->C) / support(A), range: [0, 1]
      - lift(A->C) = confidence(A->C) / support(C), range: [0, inf]

    metric_threshold : float (default: 0.0)
      Minimal threshold for the evaluation metric given by the `metric` parameter,
      to decide if a rule is of interest.

    Returns
    ----------
    pandas DataFrame with columns "antecedents" and "consequents",
      and metric columns: "antecedent support", "consequent support", "support", "confidence", "lift",
      of all rules where, `metric` >= `metric_threshold`.
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

                # check the parameter metric with the parameter threshold
                metric_score = metric_dict[metric](sAC, sA, sC)
                if metric_score >= metric_threshold:
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
    rules_df,
    max_movies=7,
):
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

    # output user movie df, and recommended movie list
    return user_movie_rules_df, recommended_movies[0:max_movies]


"""
# constants for algorithms
RATING_THRESHOLD = 3
MIN_SUPPORT = 0.07
MAX_LEN = 5
METRIC = "lift"
METRIC_THRESHOLD = 1

# load data
movies_df, ratings_df, df = load_data()

# our apriori model needs data in a matrix with:
# userId is index, columns are movie tiles, and the values
# are False/True depending on whether user rated movie over some threshold.
df = df.applymap(encode_ratings, None, rating_threshold=RATING_THRESHOLD)

# generate the frequent itemsets using the apriori algorithm
freq_itemsets = apriori(df, min_support=MIN_SUPPORT, max_len=MAX_LEN)

# support: probabilty of users watching movie M1
# support(M) = (# user watchlists containing M) / (# user watchlists)
# confidence: out of total users having watched movie M1, how many have also watched movie M2
# confidence(M1 -> M2) = (# user watchlists containing M1 and M2) / (# user watchlists containing M1)
# lift: ratio of confidence and support
# lift(M1 -> M2) = confidence(M1 -> M2) / support(M2)
# high lift suggests there is some relation between the two movies and most of the
# users who have watched movie M1 are also likely to watch movie M2.
# rules are sorted by descending value of the given metric
rules_df = create_association_rules(
    freq_itemsets, metric=METRIC, metric_threshold=METRIC_THRESHOLD
)

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
