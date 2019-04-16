import pickle
import numpy as np
from collections import Counter

def build_vocab_oneword():
    vocab_f = {}
    f = open('sgns.renmin.word','r',encoding='utf-8')
    for i in f:
        if i[1] == ' ':
            char = i[0]
            vec = i[2:].strip().split(' ')
            for i in range(300):
                vec[i] = float(vec[i])
            vocab_f[char] = vec
    f.close()
    with open('vocab_oneword.pkl','wb')as f:
        pickle.dump(vocab_f,f)

def build_vocab_stop():
    with open('vocab_oneword.pkl','rb') as f:
        vocab_f = pickle.load(f)
    vocab_stop=[]
    with open('cnews\\cnews.train.txt','r',encoding='utf-8') as f:
        for i in f:
            _, content = i.split(sep='\t')
            for j in content:
                if j not in vocab_f and j not in vocab_stop:
                    vocab_stop.append(j)
    vocab_stop = dict(Counter(vocab_stop))
    with open('vocab_stop.pkl','wb') as f:
        pickle.dump(vocab_stop,f)

def build_vocab():
    with open('vocab_stop.pkl','rb')as f:
        vocab_stop = pickle.load(f)
    vocab = {}
    with open('cnews\\cnews.train.txt','r',encoding='utf-8') as f:
        n = 0
        for i in f:
            _, content = i.split(sep='\t')
            for j in content:
                if j not in vocab_stop and j not in vocab:
                    vocab[j] = n
                    n += 1
    with open('vocab.pkl','wb')as f:
        pickle.dump(vocab,f)

def build_embedding_np():
    with open('vocab.pkl','rb')as f:
        vocab = pickle.load(f)
    with open('vocab_oneword.pkl','rb')as f:
        vocab_oneword = pickle.load(f)
    embedding = []
    keys = vocab.keys()
    for i in keys:
        embedding.append(vocab_oneword[i])
    embedding = np.array(embedding)
    with open('embedding.pkl','wb') as f:
        pickle.dump(embedding,f)

build_vocab_oneword()
build_vocab_stop()
build_vocab()
build_embedding_np()
