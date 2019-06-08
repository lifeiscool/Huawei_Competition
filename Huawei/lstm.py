import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import csv

rnn_unit = 10  # 隐层神经元的个数
lstm_layers = 2  # 隐层层数
input_size = 2
output_size = 1
lr = 0.0006  # 学习率
test_index = 0


def preprocession(file_name='wyj_csv/price_withdate.csv', date_index=0, start_column_index = 1, end_column_index = 4 ):
    """
    预处理数据，从csv文件中读取需要的属性列，返回list
    :param file_name:   csv文件名
    :param date_index:  日期列下标
    :param start_column_index:  需要的属性开始下标
    :param end_column_index:    需要的属性结束下标
    :return: origin_data:   list类型存储start_column_index：end_column_index的值
    """
    global test_index
    f = open(file_name)
    df = pd.read_csv(f)

    date = df.iloc[:, date_index].values
    for i in range(len(date)):
        if date[i][0:4] == '2018':
            test_index = i
            break

    origin_data = df.iloc[:, start_column_index:end_column_index].values
    origin_data = origin_data.tolist()
    for i in range(len(origin_data) - 1):
        origin_data[i].append(origin_data[i + 1][1])
    origin_data[i + 1].append(origin_data[i + 1][1])
    return origin_data
origin_data = preprocession()
# print("ori",len(origin_data))

test_index = 200
def get_train_data(batch_size=60, time_step=5, train_begin=0,train_end=test_index):
    batch_index = []
    train_x, train_y = [], []

    data_train = origin_data[train_begin:train_end]
    print(len(data_train))
    normalized_train_data = (data_train - np.mean(data_train, axis=0)) / np.std(data_train, axis=0)

    for i in range(len(normalized_train_data) - time_step + 1):
        if i % batch_size == 0:
            batch_index.append(i)
        train_x.append(normalized_train_data[i:i + time_step, :input_size].tolist())
        train_y.append(normalized_train_data[i:i + time_step, input_size, np.newaxis].tolist())
    batch_index.append(i+1)
    # print("train",len(train_x))
    return batch_index, train_x, train_y


# get_train_data()



def get_test_data(time_step=5, test_begin=test_index):
    # print("testIndex",test_index)
    data_test = origin_data[test_begin:]
    mean = np.mean(data_test, axis=0)
    std = np.std(data_test, axis=0)
    normalized_test_data = (data_test - mean) / std  # 标准化
    #normalized_train_data = (data_test - np.mean(data_test, axis=0)) / np.std(data_test, axis=0)  # 标准化
    #print(np.array(normalized_test_data).shape)
    size = (len(normalized_test_data) + time_step - 1) // time_step  # 有size个sample,   len 196.    size  40
    test_x, test_y = [], []
    for i in range(size - 1):
        x = normalized_test_data[i * time_step:(i + 1) * time_step, :input_size]
        y = normalized_test_data[i * time_step:(i + 1) * time_step, input_size]
        test_x.append(x.tolist())
        test_y.extend(y)
    test_x.append((normalized_test_data[(i + 1) * time_step:, :input_size]).tolist())
    test_y.extend((normalized_test_data[(i + 1) * time_step:, input_size]).tolist())
    print("test",len(test_x))
    return mean, std, test_x, test_y

get_test_data()

# ——————————————————定义神经网络变量——————————————————
# 输入层、输出层权重、偏置、dropout参数

weights = {
    'in': tf.Variable(tf.random_normal([input_size, rnn_unit])),
    'out': tf.Variable(tf.random_normal([rnn_unit, 1]))
}
biases = {
    'in': tf.Variable(tf.constant(0.1, shape=[rnn_unit, ])),
    'out': tf.Variable(tf.constant(0.1, shape=[1, ]))
}
keep_prob = tf.placeholder(tf.float32, name='keep_prob')


# ——————————————————定义神经网络变量——————————————————
def lstmCell():
    # basicLstm单元
    basicLstm = tf.nn.rnn_cell.BasicLSTMCell(rnn_unit)
    # dropout
    drop = tf.nn.rnn_cell.DropoutWrapper(basicLstm, output_keep_prob=keep_prob)
    return basicLstm


def lstm(X):
    batch_size = tf.shape(X)[0]
    time_step = tf.shape(X)[1]
    w_in = weights['in']
    b_in = biases['in']
    input = tf.reshape(X, [-1, input_size])  # 需要将tensor转成2维进行计算，计算后的结果作为隐藏层的输入
    input_rnn = tf.matmul(input, w_in) + b_in
    input_rnn = tf.reshape(input_rnn, [-1, time_step, rnn_unit])  # 将tensor转成3维，作为lstm cell的输入
    cell = tf.nn.rnn_cell.MultiRNNCell([lstmCell() for i in range(lstm_layers)])
    init_state = cell.zero_state(batch_size, dtype=tf.float32)
    output_rnn, final_states = tf.nn.dynamic_rnn(cell, input_rnn, initial_state=init_state, dtype=tf.float32)
    output = tf.reshape(output_rnn, [-1, rnn_unit])
    print(output)
    w_out = weights['out']
    b_out = biases['out']
    pred = tf.matmul(output, w_out) + b_out
    print(pred)
    return pred, final_states


# ————————————————训练模型————————————————————
losssss = []
def train_lstm(train_time,batch_size=60, time_step=5, train_begin=0):
    global losssss
    print(test_index)
    X = tf.placeholder(tf.float32, shape=[None, time_step, input_size])
    Y = tf.placeholder(tf.float32, shape=[None, time_step, output_size])
    batch_index, train_x, train_y = get_train_data(batch_size, time_step, train_begin, test_index)
    with tf.variable_scope("sec_lstm"):
        pred, _ = lstm(X)
    loss = tf.reduce_mean(tf.square(tf.reshape(pred, [-1]) - tf.reshape(Y, [-1])))
    train_op = tf.train.AdamOptimizer(lr).minimize(loss)
    saver = tf.train.Saver(tf.global_variables(), max_to_keep=15)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for i in range(train_time):  # 这个迭代次数，可以更改，越大预测效果会更好，但需要更长时间
            for step in range(len(batch_index) - 1):
                _, loss_ = sess.run([train_op, loss], feed_dict={X: train_x[batch_index[step]:batch_index[step + 1]],
                                                                 Y: train_y[batch_index[step]:batch_index[step + 1]],
                                                                 keep_prob: 0.5})
            print("Number of iterations:", i, " loss:", loss_)
            losssss.append(loss_)
        print("model_save: ", saver.save(sess, 'model_save2\\modleMutil.ckpt'))
        # I run the code on windows 10,so use  'model_save2\\modle.ckpt'
        # if you run it on Linux,please use  'model_save2/modle.ckpt'
        print("The train has finished")

yyy = []
tyyy = []
# ————————————————预测模型————————————————————
def prediction(train_time,time_step=5):
    global tyyy
    global yyy
    X = tf.placeholder(tf.float32, shape=[None, time_step, input_size])
    mean, std, test_x, test_y = get_test_data(time_step,test_index)
    print(np.array(test_x).shape)
    with tf.variable_scope("sec_lstm", reuse=tf.AUTO_REUSE):
        pred, _ = lstm(X)
    saver = tf.train.Saver(tf.global_variables())
    with tf.Session() as sess:
        # 参数恢复
        module_file = tf.train.latest_checkpoint('model_save2')
        saver.restore(sess, module_file)
        test_predict = []
        print(len(test_x))
        for step in range(len(test_x)):
            prob = sess.run(pred, feed_dict={X: [test_x[step]], keep_prob: 1})
            print(np.array(prob).shape)
            predict = prob.reshape((-1))
            test_predict.extend(predict)

        test_y = np.array(test_y) * std[1] + mean[1]
        test_predict = np.array(test_predict) * std[1] + mean[1]
        acc = np.average(np.abs(test_predict - test_y[:len(test_predict)]) / test_y[:len(test_predict)])  # 偏差程度
        print("The accuracy of this predict:", acc)
        # 以折线图表示结果
        # plt.figure()
        # origin_data = preprocession()
        # plt.plot(origin_data, color='r')
        # plt.plot([None for _ in range(test_index)]+[x for x in test_predict], color='b')
        # plt.legend(["origin data", "prediction result"])
        # plt.savefig('totalMutil.png')
        # plt.show()

        yyy = test_y
        tyyy = test_predict
        plt.figure()
        plt.plot(test_y, color='r')
        plt.plot(test_predict, color='b')
        plt.legend(["origin data", "prediction result"])
        sss = 'detailMutil'+str(train_time)+"-"+str(acc)+'.png'
        # plt.savefig(sss)
        plt.show()

train_time =450
train_lstm(train_time)
prediction(train_time)

plt.figure()
plt.plot(losssss, color='g')
plt.legend(["loss"])
sss = 'losssss.png'
# plt.savefig(sss)
plt.show()


arr0 = []
for i in range(0,len(yyy)):
    arr0.append(0)
plt.figure()
plt.plot(yyy-tyyy, color='g')
plt.plot(arr0, color='r',linestyle="-")
plt.legend(["res","0"])
sss = 'lossres.png'
# plt.savefig(sss)
plt.show()