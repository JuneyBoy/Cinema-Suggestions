import pandas as pd
import streamlit as st

import apriori

st.markdown(
    (
        "### Enter a movie and Cinema Suggestions will show you 10 "
        "of the most similar movies based movies that other users liked "
        "when watching that movie, using the Apriori Algorithm"
    )
)

# only needs to do this one time, so adding the cache statement will make it so this function only runs the first time the page is loaded but doesn't run again as long as the user doesn't close the tab in their browser
@st.cache
def load_data():
    # load the data
    ratings_df = pd.read_csv("Data_Files/ratings_small.csv")
    movies_df = pd.read_csv("Data_Files/movies_filtered.csv")

    # remove movies with blank titles
    title_mask = movies_df["title"].isna()
    movies_df = movies_df.loc[title_mask == False]

    # merge the 2 df on common column: movieId (must first convert col to int). similar to SQL Join
    movies_df = movies_df.astype({"id": "int64"})
    df = pd.merge(
        ratings_df, movies_df[["id", "title"]], left_on="movieId", right_on="id"
    )

    # remove duplicated column (movieId), and useless column (timestamp)
    df = df.drop(["timestamp", "id"], axis=1)

    # userId is index, columns are movie tiles, and the values are the ratings
    df = df.drop_duplicates(["userId", "title"])
    df = df.pivot(index="userId", columns="title", values="rating").fillna(0)
    df = df.astype("int64")

    return (movies_df, ratings_df, df)


movies_df, ratings_df, apriori_df = load_data()

# rating threshold (0.0 - 5.0): ideally around 3.0-4.0
rating_threshold = st.slider(
    "Choose a minimum movie rating to be considered `good`", 0.0, 5.0, 3.0, 0.1
)

# min support used in apriori algorithm
min_support = st.slider(
    "Choose a minimum support value for apriori", 0.02, 1.0, 0.07, 0.01
)

# max length of itemset in association rule
max_len = st.slider("Choose the maximum length of an itemset for apriori", 1, 5, 4)

# metric: support, confidence, or lift
metric = st.selectbox(
    "Select a metric to use in determining if an association rule is of interest",
    ["support", "confidence", "lift"],
    index=2,
)

# metric threshold
metric_threshold = st.slider(
    "Select a metric threshold for the chosen metric (%s) "
    "to determine if an association rule is of interest" % metric,
    0.0,
    10.0,
    0.0,
    0.1,
)

# button to start the rule creation
start_apriori = st.button("Create association rules with apriori")

# get movie from user via dropdown menu that has all the movies from the preprocessed set
user_movie = st.selectbox(
    "Select a movie:",
    [title for title in movies_df["original_title"]],
)

# max number of recommended movies
max_movies = st.slider("Choose a maximum number of recommended movies:", 0, 10, 7, 1)

# convert from original title to the title used in the algorithm
movie_idx = movies_df.index[movies_df["original_title"] == user_movie][0]
user_movie = movies_df["title"][movie_idx]

# re-generate association rules when button is clicked
if start_apriori:
    # generate frequent itemsets using apriori
    st.markdown("Apriori Running...")
    freq_itemsets = apriori.apriori(
        df=apriori_df, min_support=min_support, max_len=max_len
    )

    # generate rules from frequent itemsets
    rules_df = apriori.create_association_rules(
        frequent_itemsets=freq_itemsets,
        metric=metric,
        metric_threshold=metric_threshold,
    ).sort_values(by=[metric], ascending=False)

    st.markdown("Apriori Completed!")


# get recommended movies from rules
user_movie_rules_df, recommended_movies = apriori.recommend_movies_apriori(
    movie_title=user_movie, rules_df=rules_df, max_movies=max_movies
)

st.markdown(
    "##### Frequent Itemsets with `rating_threshold=%f`, `min_support=%f`"
    % (rating_threshold, min_support),
)
st.dataframe(freq_itemsets)

st.markdown(
    "##### Association Rules with `metric=%s`, `metric_threshold=%f`"
    % (metric, metric_threshold)
)
st.dataframe(rules_df)

st.markdown(
    "##### Association Rules for `movie=%s` with `metric=%s`, `metric_threshold=%f`"
    % (user_movie, metric, metric_threshold)
)
st.dataframe(user_movie_rules_df)

st.markdown("##### %d Most Similar Movies to `%s`" % (max_movies, user_movie))
st.markdown(recommended_movies)
