import jieba
import pickle
import numpy as np
import warnings
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import SGDClassifier
from collections import Counter
from math import log
from build_dictionary import build_dictionary
from sklearn.externals import joblib
from sklearn import metrics

def read_dir(path):
    with open(path, encoding='utf-8') as f:
        for i in f:
            s = i.strip().split(sep='\t')
            yield s[1][:600]

def save_dictionary(train_path, num = 200000):
    s=read_dir(train_path)
    D_n = build_dictionary(s, num)
    with open('Dictionary_train_600.pkl','wb') as f:
        pickle.dump(D_n,f)

def read_data(path):
    with open(path, encoding='utf-8') as f:
        for i in f:
            s = i.strip().split(sep='\t')
            yield s[1][:600],categories[s[0]]

def batch_yield(xy_train, batch_size, D_n):
    x_train, y_train = [],[]
    for x, y in xy_train:
        x_jieba = jieba.cut(x)
        x_temp = []
        for i in x_jieba:
            if i in D_n:
                x_temp.append(i)
        x_len = len(x_temp)
        x_dic = dict(Counter(x_temp))
        x_lis = list(x_dic)
        for j in x_lis:  # 计算tf-idf
            x_dic[j] = x_dic[j]/x_len * log(2*50000/D_n[j],10)
        base = np.zeros(250000)
        for key in x_lis:  # 字词映射到字典,生成稀疏向量
            base[D_n[key]] = x_dic[key]
        x_train.append(base)
        y_train.append(y)
        if len(y_train) == batch_size:
            yield x_train, y_train
            x_train, y_train = [],[]
    if y_train:
        yield x_train, y_train

def train(train_path, D_n):
    xy_train = read_data(train_path)
##    model = MultinomialNB()
    model = SGDClassifier()
    train_data = batch_yield(xy_train, 50, D_n)
    for x_train, y_train in train_data:
        model.partial_fit(x_train,y_train,classes = [0,1,2,3,4,5,6,7,8,9])
        
    joblib.dump(model, "train_model_SGD_shuffle")

def test(test_path, D_n, categories):
    xy_test = read_data(test_path)
    test_data = batch_yield(xy_test, 50, D_n)
    model = joblib.load('train_model_SGD_shuffle')
    y_pred, y_test_all = [],[],[]
    for x_test, y_test in test_data:
        y_pred.extend((model.predict(x_test)))
        y_test_all.extend(y_test)
    cm = metrics.confusion_matrix(y_test_all, y_pred)
    Precision = metrics.classification_report(y_test_all, y_pred, target_names=categories)
    return cm, Precision

categories = {'体育':0, '财经':1, '房产':2, '家居':3, '教育':4, '科技':5, '时尚':6, '时政':7, '游戏':8, '娱乐':9}

train_path = 'cnews\\cnews.train_shuffle.txt'
test_path = 'cnews\\cnews.test.txt'
warnings.filterwarnings("ignore")
with open('Dictionary_train_600.pkl','rb') as f:
    D_n = pickle.load(f)


train(train_path, D_n)
cm,Precision = test(test_path, D_n, categories)
