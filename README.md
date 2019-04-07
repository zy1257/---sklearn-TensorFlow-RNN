# ---sklearn-TensorFlow-RNN
数据集来自于http://thuctc.thunlp.org/
一共用10各类别：体育、财经、房产、家居、教育、科技、时尚、时政、游戏、娱乐
每个类别训练数据5000条，验证数据500条，测试数据1000条

基于sklearn
用jieba分词，向量空间模型VSM表示文本，tf-idf计算特征权重

贝叶斯分类器
混淆矩阵
[[975   0   0   3   1   2   1   5   4   9]
 [  0 969   9   2   0   6   0  13   1   0]
 [  0  39 803  26   8   8  10  73  14  19]
 [  0  15 301 387  28  43  24  85  85  32]
 [  2   5  11  46 777  51  18  51  17  22]
 [  0   0   3   7   1 945   9   4  28   3]
 [  0   0   0  16   2   8 959   3   4   8]
 [  0   8  35   1  12   7   0 924   3  10]
 [  1   1   2  28   1   6  25   4 916  16]
 [  0   0   0   3   3   5   6   1   5 977]]
 
     precision    recall  f1-score   support

体育       1.00      0.97      0.99      1000 
财经       0.93      0.97      0.95      1000 
房产       0.69      0.80      0.74      1000 
家居       0.75      0.39      0.51      1000 
教育       0.93      0.78      0.85      1000 
科技       0.87      0.94      0.91      1000 
时尚       0.91      0.96      0.93      1000 
时政       0.79      0.92      0.85      1000 
游戏       0.85      0.92      0.88      1000 
娱乐       0.89      0.98      0.93      1000 
  
  家居类的召回率很低，很容易被识别为房产一类
  
 随机梯度下降分类器
  混淆矩阵
[[963   1   0   1   8   1   0   1   8  17]
 [  0 993   1   0   0   0   0   6   0   0]
 [  0 113 755  17  29   6   0  24  24  32]
 [  0 210  95 466  63  17   2  27  50  70]
 [  3  18   7  16 874   8   0   2  28  44]
 [  0  34   0   7   9 905   0   1  39   5]
 [  1   9   0  35  15   0 869   2  18  51]
 [  1  53  25   5  62   3   0 790  20  41]
 [  0   2   3   9   4   5   1   2 948  26]
 [  0   0   0   2   3   0   0   0   1 994]]
 
    precision    recall  f1-score   support

体育       0.99      0.96      0.98      1000
财经       0.69      0.99      0.82      1000
房产       0.85      0.76      0.80      1000
家居       0.84      0.47      0.60      1000
教育       0.82      0.87      0.85      1000
科技       0.96      0.91      0.93      1000
时尚       1.00      0.87      0.93      1000
时政       0.92      0.79      0.85      1000
游戏       0.83      0.95      0.89      1000
娱乐       0.78      0.99      0.87      1000
对家居一类的分类效果有所提升，但仍然不够理想

基于TensorFlow RNN

混淆矩阵
[[988   0   0   0   2   6   0   2   2   0]
 [  0 976   0   0   0   1   0   7   0   0]
 [  0   1 994   2   1   1   0   1   0   0]
 [  0   9   2 857  19  50  23  29   9   2]
 [  1   7   0  11 850  52   3  42  34   0]
 [  0   0   0   3   1 982   4   3   7   0]
 [  1   2   0  24   5   9 952   0   3   4]
 [  1  15   0   1  12  14   0 953   4   0]
 [  0   1   0   1   3   4   7   2 982   0]
 
     precision    recall  f1-score   support

体育       0.99      0.99      0.99      1000
财经       0.96      0.99      0.98       984
房产       1.00      0.99      1.00      1000
家居       0.95      0.86      0.90      1000
教育       0.94      0.85      0.89      1000
科技       0.87      0.98      0.92      1000
时尚       0.96      0.95      0.95      1000
时政       0.92      0.95      0.93      1000
游戏       0.93      0.98      0.96      1000
娱乐       0.99      0.94      0.97      1000
效果得到明显提升，而且模型依然有优化的空间，后续会采用CNN来做，对比一下效果。
