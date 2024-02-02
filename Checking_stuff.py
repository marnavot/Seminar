import os
from gensim.models import Word2Vec
from gensim.test.utils import datapath
from scipy.stats import spearmanr, pearsonr
import matplotlib.pyplot as plt

model = Word2Vec.load("/cs/labs/oabend/tomer.navot/decade_models/0/1980-1989_model.model")
print(f"loaded model: {model.wv.vocab}")