from gensim.models import Word2Vec
import os
import re
import pandas as pd
import string
import chardet

import random

random.seed(42)

# step 1: count appearances of nouns, verbs, adjectives and adverbs and make a list of 1000
# most common lemmas of each category in Corpus

# # Create an empty dictionary to store lemma-based noun counts
# lemma_noun_counts = {}
# lemma_verb_counts = {}
# lemma_adj_counts = {}
# lemma_adv_counts = {}
#
#
# # Specify the folder where your text files are located
# folder_path = 'C:\\Users\\Tomer\\PycharmProjects\\pythonProject\\wordLemPoS'
# # Iterate through all files in the folder
# for filename in os.listdir(folder_path):
#     if filename.endswith(".txt"):
#         file_path = os.path.join(folder_path, filename)
#         with open(file_path, 'r', encoding='utf-8') as file:
#             lines = file.readlines()
#             for line in lines:
#                 parts = line.strip().split('\t')
#                 if len(parts) == 3 and re.search("^[a-zA-Z]", parts[1]) is not None:
#                     lemma = parts[1].lower()  # Convert the lemma to lowercase
#                     if parts[2].startswith('n'):
#                         # Check if the third column (PoS) starts with 'n' (indicating a noun)
#                         if lemma in lemma_noun_counts:
#                             lemma_noun_counts[lemma] += 1
#                         else:
#                             lemma_noun_counts[lemma] = 1
#
#                     if parts[2].startswith('v'):
#                         # Check if the third column (PoS) starts with 'v' (indicating a verb)
#                         if lemma in lemma_verb_counts:
#                             lemma_verb_counts[lemma] += 1
#                         else:
#                             lemma_verb_counts[lemma] = 1
#
#                     if parts[2].startswith('j'):
#                         # Check if the third column (PoS) starts with 'j' (indicating a adjective)
#                         if lemma in lemma_adj_counts:
#                             lemma_adj_counts[lemma] += 1
#                         else:
#                             lemma_adj_counts[lemma] = 1
#
#                     if parts[2].startswith('r'):
#                         # Check if the third column (PoS) starts with 'r' (indicating a adverb)
#                         if lemma in lemma_adv_counts:
#                             lemma_adv_counts[lemma] += 1
#                         else:
#                             lemma_adv_counts[lemma] = 1
#
#
#
# # Sort the lemma-based counts dictionary by the number of occurrences in descending order
# sorted_lemma_noun_counts = dict(sorted(lemma_noun_counts.items(), key=lambda item: item[1], reverse=True))
# sorted_lemma_verb_counts = dict(sorted(lemma_verb_counts.items(), key=lambda item: item[1], reverse=True))
# sorted_lemma_adj_counts = dict(sorted(lemma_adj_counts.items(), key=lambda item: item[1], reverse=True))
# sorted_lemma_adv_counts = dict(sorted(lemma_adv_counts.items(), key=lambda item: item[1], reverse=True))
#
#
# # Create a DataFrame for each sorted dictionary
# noun_df = pd.DataFrame({'Lemma': list(sorted_lemma_noun_counts.keys()), 'Count': list(sorted_lemma_noun_counts.values())})
# verb_df = pd.DataFrame({'Lemma': list(sorted_lemma_verb_counts.keys()), 'Count': list(sorted_lemma_verb_counts.values())})
# adj_df = pd.DataFrame({'Lemma': list(sorted_lemma_adj_counts.keys()), 'Count': list(sorted_lemma_adj_counts.values())})
# adv_df = pd.DataFrame({'Lemma': list(sorted_lemma_adv_counts.keys()), 'Count': list(sorted_lemma_adv_counts.values())})
#
# # Display the DataFrame
# noun_df.to_csv("noun_counts.csv")
# verb_df.to_csv("verb_counts.csv")
# adj_df.to_csv("adj_counts.csv")
# adv_df.to_csv("adv_counts.csv")

# Path to the directory containing your text files
# corpus_path = 'C:\\Users\\Tomer\\Documents\\עבודה סמינריונית\\wordLemPoS'
corpus_path = '/cs/usr/tomer.navot/Word_lemma_PoS'

# Set the path to save Word2Vec models
model_save_path = '/cs/labs/oabend/tomer.navot/year_models_final/'

# function to split into separate sentences, and remove punctuation and other marks
def split_into_sentences(lemmas):
    sentences = []
    current_sentence = []

    for lemma in lemmas:
        if lemma in [".", "?", "!"]:
            if current_sentence:
                sentences.append(current_sentence)
                current_sentence = []
        else:
            if lemma not in string.punctuation and lemma not in ["##", '\x00']:
                current_sentence.append(lemma)

    # Append the last sentence if it exists
    if current_sentence:
        sentences.append(current_sentence)

    return sentences

# Function to read and preprocess the content of a file
def read_file(file_path):
    with open(file_path, 'rb') as file:
        # Detect the encoding of the file
        raw_data = file.read()
        result = chardet.detect(raw_data)
        encoding = result['encoding']

    with open(file_path, 'r', encoding=encoding, errors='replace') as file:
        lines = file.readlines()
        # Extract lemma from the second column
        words = [line.split('\t')[1].lower() for line in lines if
                 len(line.split('\t')) > 1 and line.strip() and line.split('\t')[1].lower()]
        words = split_into_sentences(words)

        return words

last_available_year = 1849
# check last model created
for year in range(1850, 2010):
    year_model_path = os.path.join(model_save_path, f'{year}_model.model')
    if os.path.exists(year_model_path):
        print(f"model of {year} exists")
        last_available_year = year
    else:
        break
print(f"last available year is {last_available_year}")

# If the model for the last available year already exists, load it; otherwise, create a new model
last_available_path = os.path.join(model_save_path, f'{last_available_year}_model.model')
if os.path.exists(last_available_path):
    model = Word2Vec.load(last_available_path)
else:
    model = Word2Vec(window=5, min_count=50, workers=6)

# Loop through the years and train the Word2Vec model incrementally
for year in range(last_available_year + 1, 2010):  # Adjust the range based on your available years
    print(f"model of {year} starting")
    year_model_path = os.path.join(model_save_path, f'{year}_model.model')

    # read files for current year
    year_files = [os.path.join(corpus_path, file) for file in os.listdir(corpus_path) if f"_{year}_" in file]

    # Read the files for the current year
    sentences = [read_file(file) for file in year_files if os.path.isfile(file)]
    # flatten the sentences list and shuffle their order
    sentences = [inner_list for file_lists in sentences for inner_list in file_lists]
    random.shuffle(sentences)
    # print(sentences[:10])

    # Check if the model has an existing vocabulary
    if model.wv.key_to_index:
        # If the model has an existing vocabulary, continue training
        model.build_vocab(sentences, update=True)
        model.train(sentences, total_examples=model.corpus_count, epochs=1)
    else:
        # If the model has no prior vocabulary, build the vocabulary from scratch
        model.build_vocab(sentences)

    # Save the model for the current year
    model.save(year_model_path)
    print(f"model of {year} saved")
