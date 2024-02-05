import os
# import pandas as pd
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

    counts_dict = {"n":sorted_lemma_noun_counts, "v":sorted_lemma_verb_counts,
                   "adj":sorted_lemma_adj_counts, "adv":sorted_lemma_adv_counts}
    pickle.dump(counts_dict, open("/cs/labs/oabend/tomer.navot/counts_dict.p", "wb"))
    return counts_dict

# make_counts_dict(corpus_path)

print("loading counts")
loaded_counts_dict = pickle.load(open("/cs/labs/oabend/tomer.navot/counts_dict.p", "rb"))
print("loaded counts")

# function to load all models from a folder
def load_folder_models(folder):
    print(f"loading models from {folder}")
    models_dict = {int(file[:4]): Word2Vec.load(f"{folder}/{file}") for file in os.listdir(folder) if file.endswith(".model")}
    print(f"loaded models from {folder}")
    return models_dict


year_models_folder = "/cs/labs/oabend/tomer.navot/year_models"
year_models = load_folder_models(year_models_folder)


# Function to get vectors for lemmas
def get_vectors(lemma, models_dict):
    models = {year:model for year,model in sorted(models_dict.items())}
    return {year:model.wv[lemma] if lemma in model.wv else None for year,model in models}


man_vectors = get_vectors("man", year_models)
print("man vectors:")
print(man_vectors)

# Function to calculate cosine similarity between vectors
def calculate_cosine_similarity(lemma,models_dict):
    vectors = get_vectors(lemma, models_dict)
    cosine_sim = {}
    for i in sorted(vectors.keys())[:-1]:
        if vectors[i] is None or vectors[i + 1] is None:
            cosine_sim[i] = None
        else:
            cosine_sim[i] = cosine_similarity([vectors[i]], [vectors[i + 1]])[0, 0]
    return cosine_sim

man_cosine_similarity = calculate_cosine_similarity("man", year_models)
print("man cosine similarity:")
print(man_cosine_similarity)


# def get_n_top_lemmas(n):
#     top_by_pos = {}
#     # Loop through each PoS
#     for pos in ['noun', 'verb', 'adj', 'adv']:  # Add more PoS as needed
#         # Load PoS count data
#         pos_count_file = os.path.join(pos_counts_path, f'{pos}_counts.csv')
#         pos_data = pd.read_csv(pos_count_file)
#
#         # Extract top n lemmas
#         top_lemmas = pos_data.head(n)['Lemma'].tolist()
#         top_by_pos[pos] = top_lemmas
#
#     return top_by_pos
#
# top_100 = get_n_top_lemmas(100)
#
#
# def get_vector_dict(lemmas_by_pos, models = year_models):
#     vector_dict = {}
#     lemma_vectors = False
#     # Loop through each pos and lemma
#     for pos, lemmas in lemmas_by_pos.items():
#         for lemma in lemmas:
#             # Get vectors for the lemma
#             lemma_vectors = get_vectors(lemma, models)
#             vector_dict[(lemma, pos)] = lemma_vectors
#
#     return vector_dict
#
#
# top_100_vectors = get_vector_dict(top_100)


