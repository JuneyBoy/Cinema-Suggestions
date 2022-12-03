import streamlit as st

import apriori

st.markdown(
    (
        "### Enter a movie and Cinema Suggestions will show you 10 "
        "of the most similar movies based movies that other users liked "
        "when also liking that movie, using the Apriori Algorithm"
    )
)

# only needs to do this one time, so adding the cache statement will make it so this function only runs the first time the page is loaded but doesn't run again as long as the user doesn't close the tab in their browser
@st.cache
def load_data():
    return apriori.data_preprocessing()


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
user_movie_original = st.selectbox(
    "Select a movie:",
    [title for title in movies_df["original_title"]],
)

# max number of recommended movies
max_movies = st.slider("Choose a maximum number of recommended movies:", 0, 10, 7, 1)

# button to get recommended movies
start_recommend = st.button("Find recommendations")

# convert from original title to the title used in the algorithm
movie_idx = movies_df.index[movies_df["original_title"] == user_movie_original][0]
user_movie = movies_df["title"][movie_idx]

# re-generate association rules when button is clicked
if start_apriori:
    # encode the ratings in the df based on user input rating threshold
    apriori_df = apriori_df.applymap(
        apriori.encode_ratings, None, rating_threshold=rating_threshold
    )

    # generate frequent itemsets using apriori
    st.session_state.freq_itemsets = apriori.apriori(
        df=apriori_df, min_support=min_support, max_len=max_len
    )

    # generate rules from frequent itemsets
    st.session_state.rules_df = apriori.create_association_rules(
        frequent_itemsets=st.session_state.freq_itemsets,
        metric=metric,
        metric_threshold=metric_threshold,
    ).sort_values(by=[metric], ascending=False)

if "rules_df" in st.session_state and start_recommend:
    # get recommended movies from rules
    (
        st.session_state.user_movie_rules_df,
        st.session_state.recommended_movies,
    ) = apriori.recommend_movies_apriori(
        movie_title=user_movie,
        rules_df=st.session_state.rules_df,
        max_movies=max_movies,
    )

if "freq_itemsets" in st.session_state and "rules_df" in st.session_state:
    st.markdown(
        "##### Frequent Itemsets with `rating_threshold=%0.2f`, `min_support=%0.2f`"
        % (rating_threshold, min_support),
    )
    st.dataframe(st.session_state.freq_itemsets)

    st.markdown(
        "##### Association Rules with `metric=%s`, `metric_threshold=%0.2f`"
        % (metric, metric_threshold)
    )
    st.dataframe(st.session_state.rules_df)

if (
    "user_movie_rules_df" in st.session_state
    and "recommended_movies" in st.session_state
):
    st.markdown(
        "##### Association Rules for `movie=%s` with `metric=%s`, `metric_threshold=%0.2f`"
        % (user_movie, metric, metric_threshold)
    )
    st.dataframe(st.session_state.user_movie_rules_df)

    st.markdown(
        "##### Found `%d` recommendations based off users who liked `%s`"
        % (len(st.session_state.recommended_movies), user_movie_original)
    )

    st.markdown(st.session_state.recommended_movies)
    for movie in st.session_state.recommended_movies:
        st.markdown("`%s`" % movie)
