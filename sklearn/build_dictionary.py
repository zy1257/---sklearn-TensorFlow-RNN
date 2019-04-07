# 要求文件夹中只有文件，文件内容不需要再做处理
# 文件内容若需要处理，可修改read_dir
import jieba
import pickle
from collections import Counter
import os

def read_dir(url):
    for j in url:
        fileNames = os.listdir(j)
        for i in fileNames:
            with open(j+'\\'+i, encoding='utf-8') as f:
                s = f.read().strip()
                yield s
            
def stop_build():
    stop = {}
    with open('.\\中文停用词表.txt', encoding = 'utf-8') as f:
        n = 0
        for i in f:
            stop[i.strip()] = n
            n += 1
        stop[' ' ] = -1
        stop['\n'] = -2
    with open('.\\中文停用词表.pkl', 'wb') as f:
        pickle.dump(stop, f)

def build_dictionary(s, most_common=0):
    D_n = {}
    with open('.\\中文停用词表.pkl', 'rb') as f:
        stop = pickle.load(f)
    for j in s:
        s_jieba = jieba.cut(j)
        temp = []
        for i in s_jieba:
            if i not in temp and i not in stop:
                temp.append(i)
        for i in temp:
            if i not in D_n:
                D_n[i] = 1
            elif i in D_n:
                D_n[i] += 1
    if most_common:
        D_n = dict(Counter(D_n).most_common(most_common))
    else:
        D_n = dict(Counter(D_n))
    return D_n

if __name__ == '__main__':
    url = ['..\\..\\情感分析\\统计自然语言处理\\2000\\neg','..\\..\\情感分析\\统计自然语言处理\\2000\\pos']
    s = read_dir(url)
    D_n = build_dictionary(s,20)
