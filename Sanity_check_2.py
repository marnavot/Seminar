from gensim.models import Word2Vec
from gensim.test.utils import datapath
from scipy.stats import spearmanr

# Load Word2Vec models
def load_model(year):
    model_path = f'/cs/labs/oabend/tomer.navot/year_models/{year}_model.model'
    return Word2Vec.load(model_path)

# Evaluate models
def evaluate_models(models):
    correlations = []
    for year, model in models.items():
        # Use Gensim's evaluate_word_pairs
        result = model.evaluate_word_pairs(datapath('wordsim353.tsv'))

        # Extract Spearman correlation
        correlation = result['spearmanr'][0]
        correlations.append(correlation)

    return correlations

# Example: Evaluate models from 1810 to 2010
models = {year: load_model(year) for year in range(1810, 2011)}
correlations = evaluate_models(models)

print("Spearman correlation for each year:")
for year, correlation in zip(range(1810, 2011), correlations):
    print(f"{year}: {correlation}")
