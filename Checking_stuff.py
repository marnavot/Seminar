import os
from gensim.models import Word2Vec
from gensim.test.utils import datapath
from scipy.stats import spearmanr, pearsonr
import matplotlib.pyplot as plt

folder = "/cs/labs/oabend/tomer.navot/decade_models/0/"
for filename in os.listdir(folder):
    model = Word2Vec.load(folder + filename)
    print(f"loaded model {filename}")
    print(model.wv.most_similar("man"))
