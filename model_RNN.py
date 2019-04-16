import tensorflow.contrib.keras as kr
import tensorflow as tf
import numpy as np
import pickle
from sklearn import metrics


class RNN_Model():
    def __init__(self, train_dir,test_dir,val_dir,vocab,embedding):
        self.train_dir = train_dir
        self.test_dir = test_dir
        self.val_dir = val_dir
        self.save_path = '.\\model/RNN.model'
        self.dropout = 0.8
        self.batch_size = 64
        self.max_length = 600
        self.lr = 0.001
        self.cell_num = 128
        self.epoch = 1
        self.embedding = embedding
        self.vocab = vocab
        self.categories = {'体育': 0, '财经': 1, '房产': 2, '家居': 3, '教育': 4,
                          '科技': 5, '时尚': 6, '时政': 7, '游戏': 8, '娱乐': 9}
        self.cat_num = len(self.categories)
        self.summary_path = '.\\tensorboard'
    
    def read_file(self, path):
        with open(path, encoding = 'utf=8') as f:
            for line in f:
                label, content = line.strip().split('\t')
                if content:
                    # content=='东方稳健回报债券基金发行....'
                    yield content, self.categories[label]

    def process_file(self, path):
        content_label = self.read_file(path)
        data_id, label_id =[],[]
        for content, label in content_label:
            data_temp = []
            for i in content:
                if i in self.vocab:
                    data_temp.append(self.vocab[i])
            data_id.append(data_temp)
            label_id.append(label)
            if len(label_id) == self.batch_size:
                x_pad = kr.preprocessing.sequence.pad_sequences(data_id, self.max_length)
                y_pad = kr.utils.to_categorical(label_id, num_classes=10)  # 将标签转换为one-hot表示
                # x_pad shape:(batch_size, max_length)  type:list 内容为每个字符的编号
                # y_pad shape:(bacch_size, 10)  type:list 内容为标签编号
                yield x_pad, y_pad
                data_id, label_id =[],[]

    def build_graph(self):
        self.input_x = tf.placeholder(tf.int32, [self.batch_size, self.max_length], name='input_x')
        self.input_y = tf.placeholder(tf.float32, [self.batch_size, self.cat_num], name='input_y')
        self.keep_prob = tf.placeholder(tf.float32, name='keep_prob')

        embedding = tf.Variable(self.embedding)
        embedding_inputs = tf.nn.embedding_lookup(embedding, self.input_x)

        cells = []
        for _ in range(2):
            cell = tf.contrib.rnn.GRUCell(self.cell_num)
            cell = tf.contrib.rnn.DropoutWrapper(cell, output_keep_prob=self.dropout)
            cells.append(cell)
        rnn_cell = tf.contrib.rnn.MultiRNNCell(cells, state_is_tuple=True)
        _outputs, _ = tf.nn.dynamic_rnn(cell=rnn_cell, inputs=embedding_inputs, dtype=tf.float64)
        last = _outputs[:,-1,:]

        fc = tf.layers.dense(last, self.cell_num, name='fc1')
        fc = tf.contrib.layers.dropout(fc, self.dropout)
        logits = tf.layers.dense(fc, 10, name='fc2')
        self.y_pred = tf.argmax(tf.nn.softmax(logits), 1)

        cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=self.input_y)
        self.loss = tf.reduce_mean(cross_entropy)
        self.optim = tf.train.AdamOptimizer(learning_rate=self.lr).minimize(self.loss)

        self.input_y_cls = tf.argmax(self.input_y, 1)
        correct_pred = tf.equal(self.input_y_cls, self.y_pred)
        self.acc = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

    def train(self):
        tf.summary.scalar("loss", self.loss)
        merge = tf.summary.merge_all()
        saver = tf.train.Saver()
        sess = tf.Session()
        sess.run(tf.global_variables_initializer())
        writer = tf.summary.FileWriter(self.summary_path, sess.graph)
        best_acc = 0.0
        val_acc = []
        
        for epoch in range(self.epoch):
            print('epoch{}'.format(epoch))
            xy_train = self.process_file(self.train_dir)
            for step,(x_train, y_train) in enumerate(xy_train):
                w_merge, _ ,train_acc= sess.run([merge,self.optim,self.acc],
                                         feed_dict={self.input_x:x_train,self.input_y:y_train})
                writer.add_summary(w_merge, step)
                if train_acc > 0.92:
                    xy_val = self.process_file(self.val_dir)
                    for x_val, y_val in xy_val:
                        val_acc.append(sess.run(self.acc,
                                       feed_dict={self.input_x:x_val,self.input_y:y_val}))
                    Acc = sum(val_acc)/len(val_acc)
                    print('val_acc',Acc,'\t','train_acc',train_acc)
                    if Acc > best_acc:
                        best_acc = Acc
                        print('best_acc',best_acc)
                        saver.save(sess, save_path = self.save_path)


    def test(self):
        saver = tf.train.Saver()
        sess = tf.Session()
        sess.run(tf.global_variables_initializer())
        saver.restore(sess, self.save_path)
        xy_test = self.process_file(self.test_dir)
        y_input_cls, y_pred_cls = [],[]
        for x_test, y_test in xy_test:
            y_pred, y_input = sess.run([self.y_pred,self.input_y_cls],feed_dict={self.input_x:x_test,self.input_y:y_test})
            y_input_cls.extend(y_input)
            y_pred_cls.extend(y_pred)
        print("Precision, Recall and F1-Score...")
        print(metrics.classification_report(y_input_cls, y_pred_cls, target_names=self.categories))
        print("Confusion Matrix...")
        print(metrics.confusion_matrix(y_input_cls, y_pred_cls))
        


train_dir = '.\\cnews\\cnews.train_shuffle.txt'
test_dir = '.\\cnews\\cnews.test.txt'
val_dir = '.\\cnews\\cnews.val.txt'
with open('.\\vocab.pkl','rb') as f:
    vocab = pickle.load(f)
with open('embedding.pkl','rb') as f:
    embedding = pickle.load(f)
model = RNN_Model(train_dir,test_dir,val_dir,vocab, embedding)
model.build_graph()
model.train()
##model.test()

