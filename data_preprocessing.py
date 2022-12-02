import pandas as pd
import numpy as np
import json
import re

movies_df = pd.read_csv("Data_Files/movies_metadata.csv")
# get rid of columns that aren't being used
movies_df.drop(
    [
        "belongs_to_collection",
        "budget",
        "homepage",
        "overview",
        "popularity",
        "poster_path",
        "production_companies",
        "production_countries",
        "release_date",
        "revenue",
        "runtime",
        "spoken_languages",
        "status",
        "tagline",
        "video",
    ],
    axis=1,
    inplace=True,
)

# only keep movies that have a substantial vote count and decent overall review score
movies_df = movies_df[movies_df["vote_count"] > 300]
movies_df = movies_df[movies_df["vote_average"] >= 6.0]
# I was getting a weird error when trying to merge the metadata and credits DFs, so ensuring ID field is an int
movies_df["id"] = movies_df["id"].astype("int")

credits_df = pd.read_csv("Data_Files/credits.csv")
credits_df["id"] = credits_df["id"].astype("int")

# merge the three dfs based on IDs
movies_df = movies_df.merge(credits_df, left_on="id", right_on="id")

keywords_df = pd.read_csv("Data_Files/keywords.csv")

movies_df = movies_df.merge(keywords_df, left_on="id", right_on="id")

# 2D list, each element is a list of genres for each movie
# converting string to JSON to get each genre
# need to replace single quotes with double quotes to interpret strings as JSON
genres = [
    [genre_info["name"] for genre_info in json.loads(movie_genres.replace("'", '"'))]
    for movie_genres in movies_df["genres"]
]

movies_df["genres"] = genres

unique_genres = []

for movie_genres in genres:
    for genre in movie_genres:
        if genre not in unique_genres:
            unique_genres.append(genre)

# used to calculate cosine similarity between movies
def generate_binary(unique_list, movie_specific_list):
    binaries = np.zeros(len(unique_list))
    for item in movie_specific_list:
        binaries[unique_list.index(item)] = 1
    return binaries


genre_bin_df = {
    title: generate_binary(unique_genres, genres)
    for title, genres in zip(movies_df["original_title"], movies_df["genres"])
}


# couldn't convert to JSON easily because of some names like O'Brien having a single quote in them
# using regex instead to parse string for each actor name
# only keeping main 4 actors in each movie
casts = [
    [cast_info[9:-1] for cast_info in re.findall("'name': '[a-zA-Z\s]+'", cast)][:4]
    for cast in movies_df["cast"]
]

movies_df["cast"] = casts

unique_cast_members = []

for cast in casts:
    for cast_member in cast:
        if cast_member not in unique_cast_members:
            unique_cast_members.append(cast_member)

pd.DataFrame(unique_cast_members).to_csv("Unique_Actors_List.csv")

cast_bin_df = {
    title: generate_binary(unique_cast_members, cast)
    for title, cast in zip(movies_df["original_title"], movies_df["cast"])
}


# Parsing the crew info from the DF is a mess, can't find an easy solution to get the director so just skipping that for now and might come back to it later if we have time

# directors = [
#     [
#         crew_member["name"]
#         for crew_member in json.loads(
#             re.sub('""[a-zA-z-]+\s[a-zA-z-\']+""|None', '" "', crew).replace("'", '"')
#         )
#         if crew_member["job"] == "Director"
#     ]
#     for crew in movies_df["crew"]
# ]

# unique_directors = []

# movies_df["directors"] = directors
# for director in directors:
#     if director not in director:
#         unique_directors.append(director)

# movies_df["directors_binaries"] = movies_df["directors"].apply(
#     lambda x: generate_binary(unique_directors, x)
# )

# same issue as cast when trying to parse (ex: new year's eve)
# only keeping the top five keywords
keywords = [
    [keyword[9:-1] for keyword in re.findall("'name': '[a-zA-Z\s]+'", movie_keywords)][
        :5
    ]
    if movie_keywords
    else []
    for movie_keywords in movies_df["keywords"]
]

movies_df["keywords"] = keywords

unique_keywords = []

for movie_keywords in keywords:
    for keyword in movie_keywords:
        if keyword not in unique_keywords:
            unique_keywords.append(keyword)

keyword_bin_df = {
    title: generate_binary(unique_keywords, keywords)
    for title, keywords in zip(movies_df["original_title"], movies_df["keywords"])
}

movies_df.drop("crew", axis=1, inplace=True)

movies_df.to_csv("Data_Files/movies_filtered.csv")
pd.DataFrame(genre_bin_df).to_csv("Data_Files/genre_binaries.csv")
pd.DataFrame(cast_bin_df).to_csv("Data_Files/cast_binaries.csv")
pd.DataFrame(keyword_bin_df).to_csv("Data_Files/keyword_binaries.csv")
