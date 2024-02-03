import os
from gensim.models import Word2Vec
from gensim.test.utils import datapath
from scipy.stats import spearmanr, pearsonr
import matplotlib.pyplot as plt

folder = "/cs/labs/oabend/tomer.navot/decade_models/"
for i in range(10):
    sub_folder = f"{folder}{i}/"
    for filename in os.listdir(sub_folder):
        print(f"loading {filename}")
        model = Word2Vec.load(sub_folder + filename)
        print(f"loaded model {filename}")
        print(model.wv.most_similar("man"))
