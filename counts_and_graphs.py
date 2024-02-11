import os
import pandas as pd
# import numpy as np
# from scipy.spatial import distance
from gensim.models import Word2Vec, KeyedVectors, utils
import pickle
# import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
import math
import re
import chardet

# step 1: count appearances of nouns, verbs, adjectives and adverbs and make a list of 1000
# most common lemmas of each category in Corpus

# Specify the folder where your text files are located
corpus_path = '/cs/usr/tomer.navot/Word_lemma_PoS'

save_path = '/cs/labs/oabend/tomer.navot'


def make_full_counts_df(corpus, file_save_path):
    # Create an empty dictionary to store lemma-based noun counts
    full_df_dict = {}
    # Iterate through all files in the folder
    years = set()
    for filename in os.listdir(corpus):
        year = filename.split('_')[1]
        print(year)
        years.add(year)
        print(filename)
        file_path = os.path.join(corpus_path, filename)
        if filename.endswith(".txt"):
            with open(file_path, 'rb') as file:
                # Detect the encoding of the file
                raw_data = file.read()
                result = chardet.detect(raw_data)
                encoding = result['encoding']
            with open(file_path, 'r', encoding=encoding, errors='replace') as file:
                lines = file.readlines()
                for line in lines:
                    parts = line.strip().split('\t')
                    if len(parts) == 3 and re.search("^[a-zA-Z]", parts[1]) is not None:
                        lemma = parts[1].lower()  # Convert the lemma to lowercase
                        pos = parts[2]
                        if parts[2].startswith('n'):
                            pos = "n"
                        if parts[2].startswith('v'):
                            pos = "v"
                        if parts[2].startswith('j'):
                            pos = "adj"
                        if parts[2].startswith('r'):
                            pos = "adv"
                        if pos in ["n", "v", "adj", "adv"]:
                            lemma_id = (lemma, pos)
                            if lemma_id not in full_df_dict.keys():
                                full_df_dict[lemma_id] = {"all": 1, year: 1}
                            else:
                                full_df_dict[lemma_id]["all"] += 1
                                if year not in full_df_dict[lemma_id].keys():
                                    full_df_dict[lemma_id][year] = 1
                                else:
                                    full_df_dict[lemma_id][year] += 1
        # lemmas = [lemma_id[0]]
        # full_df = pd.DataFrame()
        print(f"finished {filename}")

    lemmas = [lemma_id[0] for lemma_id in full_df_dict.keys()]
    pos = [lemma_id[1] for lemma_id in full_df_dict.keys()]
    data = {"lemma": lemmas, "pos": pos}
    years = sorted(list(years))
    years.append("all")
    print(years)

    counts = {}
    for year in years:
        year_list = []
        for lemma in full_df_dict.keys():
            if year not in full_df_dict[lemma].keys():
                full_df_dict[lemma][year] = 0
            year_list.append(full_df_dict[lemma][year])
        counts[year] = year_list

    df_full = pd.DataFrame(data)
    df_full = df_full.assign(**counts)
    df_full.to_csv(file_save_path, encoding="utf-8", index=False)
    return df_full


# full_df_count = make_full_counts_df(corpus_path, f"{save_path}/full_count.csv")
# print(full_df_count)


def make_counts_dict(corpus):
    # Create an empty dictionary to store lemma-based noun counts
    lemma_noun_counts = {}
    lemma_verb_counts = {}
    lemma_adj_counts = {}
    lemma_adv_counts = {}

    # Iterate through all files in the folder
    for filename in os.listdir(corpus):
        file_path = os.path.join(corpus_path, filename)
        if filename.endswith(".txt"):
            with open(file_path, 'rb') as file:
                # Detect the encoding of the file
                raw_data = file.read()
                result = chardet.detect(raw_data)
                encoding = result['encoding']
            with open(file_path, 'r', encoding=encoding, errors='replace') as file:
                lines = file.readlines()
                for line in lines:
                    parts = line.strip().split('\t')
                    if len(parts) == 3 and re.search("^[a-zA-Z]", parts[1]) is not None:
                        lemma = parts[1].lower()  # Convert the lemma to lowercase
                        if parts[2].startswith('n'):
                            # Check if the third column (PoS) starts with 'n' (indicating a noun)
                            if lemma in lemma_noun_counts:
                                lemma_noun_counts[lemma] += 1
                            else:
                                lemma_noun_counts[lemma] = 1

                        if parts[2].startswith('v'):
                            # Check if the third column (PoS) starts with 'v' (indicating a verb)
                            if lemma in lemma_verb_counts:
                                lemma_verb_counts[lemma] += 1
                            else:
                                lemma_verb_counts[lemma] = 1

                        if parts[2].startswith('j'):
                            # Check if the third column (PoS) starts with 'j' (indicating a adjective)
                            if lemma in lemma_adj_counts:
                                lemma_adj_counts[lemma] += 1
                            else:
                                lemma_adj_counts[lemma] = 1

                        if parts[2].startswith('r'):
                            # Check if the third column (PoS) starts with 'r' (indicating a adverb)
                            if lemma in lemma_adv_counts:
                                lemma_adv_counts[lemma] += 1
                            else:
                                lemma_adv_counts[lemma] = 1

    # Sort the lemma-based counts dictionary by the number of occurrences in descending order
    sorted_lemma_noun_counts = dict(sorted(lemma_noun_counts.items(), key=lambda item: item[1], reverse=True))
    sorted_lemma_verb_counts = dict(sorted(lemma_verb_counts.items(), key=lambda item: item[1], reverse=True))
    sorted_lemma_adj_counts = dict(sorted(lemma_adj_counts.items(), key=lambda item: item[1], reverse=True))
    sorted_lemma_adv_counts = dict(sorted(lemma_adv_counts.items(), key=lambda item: item[1], reverse=True))

    counts_dict = {"n": sorted_lemma_noun_counts, "v": sorted_lemma_verb_counts,
                   "adj": sorted_lemma_adj_counts, "adv": sorted_lemma_adv_counts}
    pickle.dump(counts_dict, open("/cs/labs/oabend/tomer.navot/counts_dict.p", "wb"))
    return counts_dict


# make_counts_dict(corpus_path)

print("loading counts")
loaded_counts_dict = pickle.load(open("/cs/labs/oabend/tomer.navot/counts_dict.p", "rb"))
print("loaded counts")
number_of_lemmas = {key: len(loaded_counts_dict[key]) for key in loaded_counts_dict.keys()}
print(number_of_lemmas)


def get_lemmas_that_appear_more_than_n(counts, n, pos_list=None):
    if pos_list is None:
        pos_list = ["n", "v", "adj"]
    results_dict = {}
    for pos, inner_dict in counts.items():
        if pos in pos_list:
            results_dict[pos] = [lemma for lemma in inner_dict.keys() if inner_dict[lemma] >= n]

    return results_dict


# lemmas_counts_more_than_20 = get_lemmas_that_appear_more_than_n(loaded_counts_dict, n=20)
# print(lemmas_counts_more_than_20)

def get_top_n_lemmas(counts, n):
    result_dict = {}
    for pos, inner_dict in counts.items():
        top_n_lemmas = list(inner_dict.keys())[:n]
        result_dict[pos] = top_n_lemmas
    return result_dict


# function to load all models from a folder
def load_folder_models(folder):
    print(f"loading models from {folder}")
    models_dict = {int(file[:4]): Word2Vec.load(f"{folder}/{file}") for file in os.listdir(folder) if
                   file.endswith(".model")}
    print(f"loaded models from {folder}")
    return models_dict


#
# year_models_folder = "/cs/labs/oabend/tomer.navot/year_models"
# year_models = load_folder_models(year_models_folder)

# year_models_2_folder = "/cs/labs/oabend/tomer.navot/year_models_2"
# year_models_2 = load_folder_models(year_models_2_folder)


# Function to get vectors for lemmas
def get_vectors(lemma, models_dict):
    return {year: model.wv[lemma] if lemma in model.wv else None for year, model in models_dict.items()}


#
# man_vectors = get_vectors("man", year_models)
# print("man vectors:")
# print(man_vectors)

# Function to calculate cosine similarity between vectors
def calculate_cosine_similarity(lemma, models_dict):
    vectors = get_vectors(lemma, models_dict)
    cosine_sim = {}
    years = sorted(vectors.keys())
    for i in range(len(years) - 1):
        if vectors[years[i]] is None or vectors[years[i + 1]] is None:
            cosine_sim[years[i]] = None
        else:
            cosine_sim[years[i]] = cosine_similarity([vectors[years[i]]], [vectors[years[i + 1]]])[0, 0]
    return cosine_sim


def cosine_similarity_years_apart(lemma, models_dict, years_distance=10):
    vectors = get_vectors(lemma, models_dict)
    cosine_sim = {}
    years = sorted(vectors.keys())
    for i in years[:-years_distance]:
        if vectors[i] is None or vectors[i + years_distance] is None:
            cosine_sim[i] = None
        else:
            cosine_sim[i] = cosine_similarity([vectors[i]], [vectors[i + years_distance]])[0, 0]
    return cosine_sim


# top_100_lemmas = get_top_n_lemmas(loaded_counts_dict, 100)
# print(top_100_lemmas)

def all_lemmas_cosine_similarity(lemma_dict, models_dict):
    result_dict = {}
    for pos, inner_list in lemma_dict.items():
        result_dict[pos] = {lemma: calculate_cosine_similarity(lemma, models_dict) for lemma in inner_list}
    return result_dict


def all_lemmas_cosine_similarity_years_apart(lemma_dict, models_dict, years_distance=10):
    result_dict = {}
    for pos, inner_list in lemma_dict.items():
        result_dict[pos] = {lemma: cosine_similarity_years_apart(lemma, models_dict, years_distance)
                            for lemma in inner_list}
    return result_dict


# get all nouns, verbs and adjectives
# all_lemmas = get_top_n_lemmas(loaded_counts_dict, len(loaded_counts_dict))
# all_n_v_adj = {pos:inner_dict for pos, inner_dict in all_lemmas if pos in ["n", "v", "adj"]}

all_n_v_adj = {pos: inner_dict.keys() for pos, inner_dict in loaded_counts_dict.items() if pos in ["n", "v", "adj"]}

# load year models
year_models_folder = "/cs/labs/oabend/tomer.navot/year_models_final"
year_models = load_folder_models(year_models_folder)
print("loaded year models")

# calculated all vocab cosine similarity
year_models_all_cosine_sim = all_lemmas_cosine_similarity_years_apart(all_n_v_adj, year_models, years_distance=10)
print("calculated year models' cosine similarity")
year_models_organized_cosine_sim = {(lemma, pos): cosine_sim for pos, inner_dict in year_models_all_cosine_sim.items()
                                    for lemma, cosine_sim in inner_dict.items()}
print(list(year_models_organized_cosine_sim.items())[:10])

lemmas_list = [key[0] for key in year_models_organized_cosine_sim.keys()]
pos_list = [key[1] for key in year_models_organized_cosine_sim.keys()]
year_data = {"lemma": lemmas_list, "pos": pos_list}
for year in year_models_organized_cosine_sim[('man', 'n')].keys():
    year_data[f"{year}"] = [year_models_organized_cosine_sim[lemma_tup][year] for lemma_tup
                            in year_models_organized_cosine_sim.keys()]

year_cosine_sim_df = pd.DataFrame(data=year_data)
year_cosine_sim_df.to_csv("/cs/labs/oabend/tomer.navot/year_cosine_sim_final.csv", index=False)


# decade_models_folder = "/cs/labs/oabend/tomer.navot/decade_models_final"
# for i in range(10):


# # create dictionary of cosine similarity of all 100 top lemmas, with year models
# top_100_lemmas_cosine_similarity = all_lemmas_cosine_similarity(top_100_lemmas, year_models)
# pickle.dump(top_100_lemmas_cosine_similarity,

# # create dictionary of cosine similarity (10 years distance) for all lemmas that appear more than 5000 times
# lemmas_more_than_5000 = get_lemmas_that_appear_more_than_n(loaded_counts_dict, 5000)
# more_than_5000_cosine_similarity = all_lemmas_cosine_similarity_years_apart(lemmas_more_than_5000, year_models_2, 10)
# pickle.dump(more_than_5000_cosine_similarity,
#             open("/cs/labs/oabend/tomer.navot/year_models_cosine_similarity_more_than_5000.p", "wb"))

# # dictionary of cosine similarity for the model 10 years after, of all top 100 lemmas
# top_100_similarity_decades = all_lemmas_cosine_similarity_years_apart(top_100_lemmas, year_models_2, 10)
# pickle.dump(top_100_similarity_decades,
#             open('/cs/labs/oabend/tomer.navot/year_models_2_cosine_similarity_decades.p', 'wb'))


# with open("/cs/labs/oabend/tomer.navot/year_models_cosine_similarity.p", 'rb') as file:
#     # Load the dictionary from the pickle file
#     loaded_dict = pickle.load(file)
# # print(loaded_dict["gay"])

# # create dictionary of cosine similarity of all top 100 words, for each of the decade model sets
# decade_folder = "/cs/labs/oabend/tomer.navot/decade_models"
# for i in range(10):
#     bin_models_dict = load_folder_models(f"{decade_folder}/{i}")
#     bin_similarity_dict = all_lemmas_cosine_similarity(top_100_lemmas, bin_models_dict)
#     pickle.dump(bin_similarity_dict, open(f"/cs/labs/oabend/tomer.navot/decade_models_bin_{i}_cosine_similarity.p", "wb"))

# # load cosine similarity dictionaries and print an example
# for i in range(10):
#     with open(f"/cs/labs/oabend/tomer.navot/decade_models_bin_{i}_cosine_similarity.p", 'rb') as file:
#         loaded_dict = pickle.load(file)
#         print(loaded_dict["adj"]["heavy"])


# # create dict of known lemmas that underwent semantic change
# changed_lemmas = {"n": ["car","mouse", "humor", "bear", "weasel", "dog", "hound", "ass", "toilet", "dude", "disease"],
#                   "v": ["arrive"],
#                   "adj":["gay", "silly"],
#                   "adv":["terribly", "horribly", "awfully"]}
# # calculate their cosine similarity in the year models and save
# changed_lemmas_cosine = all_lemmas_cosine_similarity(changed_lemmas, year_models)
# pickle.dump(changed_lemmas_cosine, open("/cs/labs/oabend/tomer.navot/year_models_cosine_similarity_changed_lemmas.p", "wb"))
#
# # the same for all models:
# decade_folder = "/cs/labs/oabend/tomer.navot/decade_models"
# for i in range(10):
#     bin_models_dict = load_folder_models(f"{decade_folder}/{i}")
#     bin_similarity_dict = all_lemmas_cosine_similarity(changed_lemmas, bin_models_dict)
#     pickle.dump(bin_similarity_dict, open(f"/cs/labs/oabend/tomer.navot/decade_models_bin_{i}_cosine_similarity.p", "wb")


def get_top_k_similar_vectors(lemma, model, k=25):
    top_k_similar = model.wv.similar_by_word(lemma, topn=k)
    similar_vectors = {word[0]: model.wv[word[0]] for word in top_k_similar}
    return similar_vectors
#
# print(get_top_k_similar_vectors("gay", model_1990))
