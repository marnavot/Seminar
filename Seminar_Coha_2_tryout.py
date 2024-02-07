from gensim.models import Word2Vec
import os
import re
import pandas as pd
import string
import chardet
# import nltk
# nltk.download('punkt')
# from nltk.tokenize import word_tokenize
import random

random.seed(42)

# Path to the directory containing your text files
# corpus_path = 'C:\\Users\\Tomer\\Documents\\עבודה סמינריונית\\wordLemPoS'
corpus_path = '/cs/usr/tomer.navot/Word_lemma_PoS'

# Set the path to save Word2Vec models
model_save_path = '/cs/labs/oabend/tomer.navot/year_models_tryout/'

# function to split into separate sentences
def split_into_sentences(lemmas):
    sentences = []
    current_sentence = []

    for lemma in lemmas:
        if lemma in [".", "?", "!"]:
            if current_sentence:
                sentences.append(current_sentence)
                current_sentence = []
        else:
            if lemma not in string.punctuation and lemma not in ["##", '\x00'] :
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
        # words = [line.split('\t')[1].lower() for line in lines if
        #          len(line.split('\t')) > 1 and line.strip() and line.split('\t')[1].lower() not in string.punctuation]

        return words


        # text = file.read()
        # # Tokenize the text into sentences
        # sentences = nltk.sent_tokenize(text)
        # lemma_sentences = []
        # # Tokenize each sentence into words and extract lemmas
        # for sentence in sentences:
        #     words = word_tokenize(sentence)
        #     lemmas = [word.split('\t')[1].lower() for word in words if
        #               len(word.split('\t')) > 1 and word.strip() and word.split('\t')[
        #                   1].lower() not in string.punctuation]
        #     lemma_sentences.append(lemmas)
        # return lemma_sentences

last_available_year = 1899
# check last model created
for year in range(1900, 1905):
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
    model = Word2Vec(window=5, min_count=5, workers=4)

# Loop through the years and train the Word2Vec model incrementally
for year in range(last_available_year + 1, 1905):  # Adjust the range based on your available years
    print(f"model of {year} starting")
    year_model_path = os.path.join(model_save_path, f'{year}_model.model')

    # read files for current year
    year_files = [os.path.join(corpus_path, file) for file in os.listdir(corpus_path) if str(year) in file]
    print(year_files)

    # Read the files for the current year
    sentences = [read_file(file) for file in year_files if os.path.isfile(file)]
    sentences = [inner_list for file_lists in sentences for inner_list in file_lists]

    # Set a fixed seed (e.g., 42)
    random.shuffle(sentences)
    print(sentences[:10])


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

# year_lens = {}
# for year in range(1810, 2011):
#     year_files = [os.path.join(corpus_path, file) for file in os.listdir(corpus_path) if f"_{year}_" in file]
#     sentences = [read_file(file) for file in year_files if os.path.isfile(file)]
#     sentences_flat = [word for sentence in sentences for word in sentence]
#     year_lens[year] = len(sentences_flat)
#
# print(year_lens)

