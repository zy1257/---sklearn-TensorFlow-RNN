from gensim.models import Word2Vec
import pickle
import numpy as np
from collections import Counter

m = Word2Vec.load('.\\word2vec_model\\cnews_train.model')
keys = list(m.wv.vocab.keys())
vocab_size = len(keys)
embedding = []
for i in range(vocab_size):
    embedding.append(m[keys[i]])
embedding = np.array(embedding)
with open('embedding.pkl','wb') as f:
    pickle.dump(embedding,f)
del embedding


vocab = {}
for value ,key in enumerate(keys):
    vocab[key] = value

with open('vocab.pkl','wb') as f:
    pickle.dump(vocab, f)
