from gensim.models import Word2Vec
import os
import re
import pandas as pd
import string
import chardet
import pickle

import random

names = set()

# Specify the folder where your text files are located
folder_path = '/cs/usr/tomer.navot/Word_lemma_PoS'
# Iterate through all files in the folder
for filename in os.listdir(folder_path):
    if filename.endswith(".txt"):
        file_path = os.path.join(folder_path, filename)
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
                    if parts[2].startswith('np'):
                        # Check if the third column (PoS) starts with 'n' (indicating a noun)
                        names.add(lemma)

pickle.dump(names,
                 open("/cs/labs/oabend/tomer.navot/names.pkl", "wb"))
