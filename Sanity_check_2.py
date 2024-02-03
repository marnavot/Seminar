# from gensim.models import Word2Vec
# from gensim.test.utils import datapath
# from scipy.stats import spearmanr
# import os
#
# def all_folder_models(folder_path):
#     files = sorted(list(os.listdir(folder_path)))
#     return files
#
#
# def evaluate_model(path):
#     model = Word2Vec.load(path)
#     result = model.wv.evaluate_word_pairs(datapath('/cs/labs/oabend/tomer.navot/wordsim353.tsv'))
#     # print(result)
#     pearson = result[0].statistic
#     spearman = result[1].statistic
#     print(f"Pearson correlation of {path}: {pearson}")
#     print(f"Spearman correlation of {path}: {spearman}")
#     return spearman
#
# evaluate_model('/cs/labs/oabend/tomer.navot/year_models/1991_model.model')
#
# def evaluate_models(folder_path):
#     year_models_list = all_folder_models(folder_path)
#     correlations = []
#     for year_model in year_models_list:
#         path = f"{folder_path}{year_model}"
#         correlation = evaluate_model(path)
#         correlations.append(correlation)
#     return correlations
#
# year_models_path = "/cs/labs/oabend/tomer.navot/year_models/"
# year_correlations = evaluate_models(year_models_path)
# print(year_correlations)
#
# # Evaluate models
# # def evaluate_models(model):
# #     correlations = []
# #     for year, model in models.items():
# #         # Use Gensim's evaluate_word_pairs
# #         result = model.evaluate_word_pairs(datapath('wordsim353.tsv'))
# #
# #         # Extract Spearman correlation
# #         correlation = result['spearmanr'][0]
# #         correlations.append(correlation)
# #
# #     return correlations
# #
# # # Example: Evaluate models from 1810 to 2010
# # models = {year: load_model(year) for year in range(1810, 2011)}
# # correlations = evaluate_models(models)
# #
# # print("Spearman correlation for each year:")
# # for year, correlation in zip(range(1810, 2011), correlations):
# #     print(f"{year}: {correlation}")

import os
from gensim.models import Word2Vec
from gensim.test.utils import datapath
from scipy.stats import spearmanr, pearsonr
import matplotlib.pyplot as plt


def evaluate_model(path):
    model = Word2Vec.load(path)
    spearman_result = model.wv.evaluate_word_pairs(datapath('/cs/labs/oabend/tomer.navot/wordsim353.tsv'))[1].statistic
    pearson_result = model.wv.evaluate_word_pairs(datapath('/cs/labs/oabend/tomer.navot/wordsim353.tsv'))[0].statistic
    print(f"Pearson correlation of {path}: {pearson_result}")
    print(f"Spearman correlation of {path}: {spearman_result}")
    print(f"most similar words to 'man' are: {model.wv.most_similar('man')}")
    print(f"most similar words to 'gay' are: {model.wv.most_similar('gay')}")
    return spearman_result, pearson_result


def evaluate_models(folder_path):
    models_list = sorted([file for file in os.listdir(folder_path) if file.endswith('.model')])
    spearman_correlations = []
    pearson_correlations = []

    for model in models_list:
        path = os.path.join(folder_path, model)
        spearman, pearson = evaluate_model(path)
        spearman_correlations.append(spearman)
        pearson_correlations.append(pearson)

    print(f"Spearman correlations of {folder_path}: {spearman_correlations}")
    print(f"Pearson correlation of {folder_path}: {pearson_correlations}")
    return spearman_correlations, pearson_correlations


def plot_correlations(folder_path, plot_name):
    spearman_correlations, pearson_correlations = evaluate_models(folder_path)

    # Extract years from filenames (taking the first four characters)
    years = [int(model[:4]) for model in sorted(os.listdir(folder_path))]

    plt.plot(years, spearman_correlations, label='Spearman')
    plt.plot(years, pearson_correlations, label='Pearson')
    plt.xlabel('Year')
    plt.ylabel('Correlation Coefficient')
    plt.title('Word Similarity Correlations Over Time')
    plt.legend()
    plt.savefig(f'/cs/labs/oabend/tomer.navot/plots/{plot_name}.png')
    plt.show()


# Example usage:
year_models_path = "/cs/labs/oabend/tomer.navot/year_models/"
# plot_correlations(year_models_path, plot_name="year_models_correlations")
evaluate_models(year_models_path)

for i in range(10):
    models_path = f"/cs/labs/oabend/tomer.navot/decade_models/{i}/"
    evaluate_models(models_path)


