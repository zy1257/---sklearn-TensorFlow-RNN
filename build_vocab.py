from collections import Counter
import pickle
train_dir = '.\\cnews\\cnews.train.txt'

def read_vocab(vocab_dir):
    with open(vocab_dir, encoding ='utf-8') as f:
        for i in f:
            _, s = i.split(sep='\t')
            yield s

def build_vocab(train_dir):
    s = read_vocab(train_dir)
    vocab = Counter(next(s))
    for i in s:
        vocab.update(Counter(i))
    keys = vocab.most_common(5000)
    vocab = {}
    for i,(key,_) in enumerate(keys):
        vocab[key] = i
    with open('train_vocab.pkl','wb') as f:
        pickle.dump(vocab,f)

build_vocab(train_dir)
