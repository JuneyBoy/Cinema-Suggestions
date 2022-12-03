import streamlit as st

import apriori

# Use the full page instead of a narrow central column
st.set_page_config(layout="wide")

# only needs to do this one time, so adding the cache statement will make it so this function only runs the first time the page is loaded but doesn't run again as long as the user doesn't close the tab in their browser
@st.cache
def load_data():
    return apriori.data_preprocessing()


movies_df, ratings_df, apriori_df = load_data()

st.markdown(
    (
        "### Enter a movie and Cinema Suggestions will show you 10 "
        "of the most similar movies based movies that other users liked "
        "when also liking that movie, using the Apriori Algorithm"
    )
)

# split page into columns
c1, c2 = st.columns((1, 1))

# RULE GENERATION
# rating threshold (0.0 - 5.0): ideally around 3.0-4.0
# min support used in apriori algorithm
# max length of itemset in association rule
# metric: support, confidence, or lift
# metric threshold: minimum value for metric for rules of interest
METRICS = ["support", "confidence", "lift"]
c1.markdown("#### Rule Generation")
rating_threshold = c1.slider("Minimum `good` movie rating", 0.0, 5.0, 3.0, 0.1)
min_support = c1.slider("Minimum support value for apriori", 0.02, 1.0, 0.07, 0.01)
max_len = c1.slider("Maximum size of an itemset for apriori", 1, 5, 5)
metric = c1.selectbox("Metric of interest for rule generation", METRICS, index=2)
metric_threshold = c1.slider(
    "Minimum metric (%s) value for rule generation" % metric, 0.0, 10.0, 0.0, 0.1
)
start_apriori = c1.button("GENERATE ASSOCIATION RULES")

# MOVIE RECOMMENDER
# get movie from user via dropdown menu that has all the movies from the preprocessed set
# max number of recommended movies
c2.markdown("#### Recommend Movies")
user_movie_og = c2.selectbox("Movie:", [title for title in movies_df["original_title"]])
start_recommend = c2.button("FIND RECOMMENDATIONS")

if start_apriori:
    # encode the ratings in the df to True/False
    apriori_df = apriori_df.applymap(
        apriori.encode_ratings, None, rating_threshold=rating_threshold
    )

    # generate frequent itemsets
    st.session_state.freq_itemsets = apriori.apriori(
        df=apriori_df, min_support=min_support, max_len=max_len
    )

    # generate association rules
    st.session_state.rules_df = apriori.create_association_rules(
        frequent_itemsets=st.session_state.freq_itemsets,
        metric=metric,
        metric_threshold=metric_threshold,
    )

if "rules_df" in st.session_state and "freq_itemsets" in st.session_state:
    # show user
    # make copy to prevent streamlit "autofixes" (which converts frozenset to string)
    c1.markdown("#### Association Rules")
    c1.dataframe(st.session_state.rules_df.copy())

if start_recommend:
    if "rules_df" in st.session_state and "freq_itemsets" in st.session_state:
        # convert from original title to the title used in the algorithm
        movie_idx = movies_df.index[movies_df["original_title"] == user_movie_og][0]
        user_movie = movies_df["title"][movie_idx]

        # get movie recommendations
        user_movie_rules_df, movies = apriori.recommend_movies_apriori(
            movie_title=user_movie,
            rules_df=st.session_state.rules_df,
            max_movies=10,
        )

        # show user
        c2.markdown("#### Recommended Movies (`%s`)" % user_movie_og)
        if len(movies) > 0:
            for movie in movies:
                c2.markdown(" `%s`" % movie)
            # make copy to prevent streamlit "autofixes" (which converts frozenset to string)
            c2.markdown("#### Association Rules Containing `%s`" % user_movie_og)
            c2.dataframe(user_movie_rules_df.copy())
        else:
            c2.info(
                "Sorry, no recommendations could be made from the association rules "
                "with the movie `%s`...\n\r"
                "Please try another movie." % user_movie_og
            )

    else:
        c2.error("Please generate association rules first!")
