import os
# import pandas as pd
# import numpy as np
# from scipy.spatial import distance
from gensim.models import Word2Vec, KeyedVectors, utils
# import pickle
import matplotlib.pyplot as plt
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
    print(counts_dict)
    return counts_dict


make_counts_dict(corpus_path)