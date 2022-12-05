import time

import streamlit as st

import apriori

# Use the full page instead of a narrow central column
st.set_page_config(layout="wide")

# only needs to do this one time, so adding the cache statement will make it so this function only runs the first time the page is loaded but doesn't run again as long as the user doesn't close the tab in their browser
@st.cache
def load_data():
    return apriori.load_data()


# convert "antecedents" and "consequents" values in df from frozenset to list
def format_rules_df(rules_df):
    if not rules_df.shape[0]:
        return rules_df
    rules_df.loc[:, "antecedents"] = rules_df["antecedents"].apply(lambda x: list(x))
    rules_df.loc[:, "consequents"] = rules_df["consequents"].apply(lambda x: list(x))
    return rules_df


movies_df, ratings_df, ratings_filtered_df, apriori_df = load_data()

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
c1.markdown("#### Rule Generation (Apriori)")
rating_threshold = c1.slider(
    "Minimum `good` movie rating for apriori", 6.0, 10.0, 6.0, 0.1
)
min_support = c1.slider("Minimum support value for apriori", 0.02, 1.0, 0.07, 0.01)
max_k_itemsets = c1.slider("Maximum itemset size for apriori", 1, 5, 5)
metric = c1.selectbox("Metric of interest for rule generation", METRICS, index=2)
start_apriori = c1.button("GENERATE ASSOCIATION RULES")

# MOVIE RECOMMENDER
# get movie from user via dropdown menu that has all the movies from the preprocessed set
# max number of recommended movies
c2.markdown("#### Recommend Movies")
user_movie = c2.selectbox("Movie:", [title for title in movies_df["original_title"]])
max_movies = c2.slider("Maximum number of movie recommendations", 1, 10, 10, 1)
start_recommend = c2.button("FIND RECOMMENDATIONS")

if start_apriori:
    st.session_state.time_apriori_start = time.time()
    # encode the ratings in the df to True/False
    # divide threshold by 2 because ratings in df are from 0-5, instead of 0-10
    apriori_df = apriori_df.applymap(
        apriori.encode_ratings, None, rating_threshold=rating_threshold / 2
    )

    # generate frequent itemsets
    st.session_state.freq_itemsets = apriori.apriori(
        df=apriori_df, min_support=min_support, max_k_itemsets=max_k_itemsets
    )

    # generate association rules
    st.session_state.rules_df = apriori.create_association_rules(
        frequent_itemsets=st.session_state.freq_itemsets, metric=metric
    )
    st.session_state.time_apriori_end = time.time()

if "rules_df" in st.session_state and "freq_itemsets" in st.session_state:
    # show user
    # make copy to prevent streamlit "autofixes" (which converts frozenset to string)
    c1.markdown("#### Association Rules")
    c1.markdown(
        "Apriori finished execution in: `%.4f` seconds"
        % (st.session_state.time_apriori_end - st.session_state.time_apriori_start)
    )
    rules_df = format_rules_df(st.session_state.rules_df)

    # check that dataframe is within memory limits to display
    if rules_df.memory_usage(deep=True).sum() < 200 * 1e6:
        c1.dataframe(rules_df)
    else:
        c1.warning("The DataFrame was too large to display, so here's a part of it.")
        c1.dataframe(rules_df.head(1000))

if start_recommend:
    if "rules_df" in st.session_state and "freq_itemsets" in st.session_state:
        time_start = time.time()
        # get movie recommendations
        user_movie_rules_df, recommended_movies_df = apriori.recommend_movies_apriori(
            movie_title=user_movie,
            movies_df=movies_df,
            ratings_filtered_df=ratings_filtered_df,
            rules_df=st.session_state.rules_df,
            max_movies=max_movies,
        )

        # show user the recommended movies
        c2.markdown("#### Recommended Movies (`%s`)" % user_movie)
        c2.markdown(
            "Found `%d` movie recommendations in: `%.4f` seconds"
            % (recommended_movies_df.shape[0], (time.time() - time_start))
        )
        if not recommended_movies_df.empty:
            c2.table(
                recommended_movies_df.style.format(
                    {
                        "Avg Rating of All Users": "{:.2f}",
                        f"Avg Rating of Users Who Liked {user_movie}": "{:.2f}",
                    }
                )
            )

            # make copy to prevent streamlit "autofixes" (which converts frozenset to string)
            c2.markdown("#### Association Rules, where antecedent is `%s`" % user_movie)
            c2.dataframe(format_rules_df(user_movie_rules_df))
        else:
            c2.info(
                "Sorry, no recommendations could be made from the association rules "
                "with the movie `%s`...\n\r"
                "Please try another movie (one that is in the antecedent of a rule)."
                % user_movie
            )
    else:
        c2.error("Please generate association rules first!")
