from gensim.models import Word2Vec
from gensim.test.utils import datapath
from scipy.stats import spearmanr
import os

# Load Word2Vec models
# def load_model(year):
#     model_path = f'/cs/labs/oabend/tstopomer.navot/year_models/{year}_model.model'
#     return Word2Vec.load(model_path)

def all_folder_models(folder_path):
    files = list(os.listdir(folder_path)).sort()
    return files


def evaluate_model(path):
    model = Word2Vec.load(path)
    result = model.evaluate_word_pairs(datapath('wordsim535.tsv'))
    correlation = result['spearmanr'][0]
    print(f"Correlation of {path}: {correlation}")
    return correlation

def evaluate_models(folder_path):
    year_models_list = all_folder_models(folder_path)
    correlations = []
    for year_model in year_models_list:
        path = f"{folder_path}{year_model}"
        correlation = evaluate_model(path)
        correlations.append(correlation)
    return correlations

year_models_path = "/cs/labs/oabend/tomer.navot/year_models/"
year_correlations = evaluate_models(year_models_path)
print(year_correlations)

# Evaluate models
# def evaluate_models(model):
#     correlations = []
#     for year, model in models.items():
#         # Use Gensim's evaluate_word_pairs
#         result = model.evaluate_word_pairs(datapath('wordsim353.tsv'))
#
#         # Extract Spearman correlation
#         correlation = result['spearmanr'][0]
#         correlations.append(correlation)
#
#     return correlations
#
# # Example: Evaluate models from 1810 to 2010
# models = {year: load_model(year) for year in range(1810, 2011)}
# correlations = evaluate_models(models)
#
# print("Spearman correlation for each year:")
# for year, correlation in zip(range(1810, 2011), correlations):
#     print(f"{year}: {correlation}")
