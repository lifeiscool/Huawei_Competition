#通过CNN训练句向量，标签为相对前日涨跌幅，最后得到代表每条评论的特征
import csv
import os
from keras.callbacks import ModelCheckpoint
import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from keras.layers import *
from keras.models import *
from keras.optimizers import Adam
from sklearn.preprocessing import MinMaxScaler

def gety_train(pripath,textpath,labelpath):
    seq_len=1
    # 价格日期，一个日期对应一个标签
    with open(pripath, 'r') as rf:
        reader = csv.reader(rf)
        date = [row[0][5:] for row in reader]
        date.pop(0)
        print(len(date))

    # 取文件最后一列作为标签
    with open(labelpath, 'r') as rf1:
        reader = csv.reader(rf1)
        train_labels = [rs[10] for rs in reader]
        train_labels.pop(0)
        print(len(train_labels))

    # 评论日期
    with open(textpath, 'r', encoding='utf-8')as csvfile:
        r1 = csv.reader(csvfile)
        news_train_date = [row[4][0:5] for row in r1]
        news_train_date.reverse()
        print(len(news_train_date))
    # 根据时间设置训练集的标签
    a = 0
    y_train_data = []
    for B in news_train_date:
        try:
            if B == date[a]:
                y_train_data.append(train_labels[a])
            elif B == date[a + 1]:
                y_train_data.append(train_labels[a + 1])
                a += 1
            else:
                y_train_data.append(train_labels[a])
        except IndexError:
            break
    count = len(y_train_data) - 1
    for i, vec in enumerate(news_train_date, 0):
        if int(i) > count:
            y_train_data.append(train_labels[len(date) - 1])
    print(len(y_train_data))
    y_train_data.pop(0)
    y_train_data=np.array(y_train_data)
    # print(y_train_data.shape)
    # y_train = np.array([y_train_data[i + seq_len, 0] for i in range(y_train_data.shape[0] - seq_len)])
    return y_train_data

def getX_train(path):
    seq_len=1
    # 句向量文件作为训练集样本
    x_train_data = []
    with open(path, 'r', encoding='utf-8')as nf:
        r2 = csv.reader(nf)
        for val in r2:
            x_train_data.append(val)
    print(len(x_train_data))
    x_train_data=np.array(x_train_data)
    x_train = np.array([x_train_data[i: i + seq_len, :] for i in range(x_train_data.shape[0] - seq_len)])
    return  x_train

def getX_test(path):
    seq_len = 1
    # 句向量文件作为训练集样本
    x_test_data = []
    with open(path, 'r', encoding='utf-8')as nf:
        r2 = csv.reader(nf)
        for val in r2:
            x_test_data.append(val)
    x_test_data = np.array(x_test_data)
    x_test = np.array([x_test_data[i: i + seq_len, :] for i in range(x_test_data.shape[0] - seq_len)])
    return x_test

def gety_test(pripath,textpath,labelpath):
    # 价格日期，一个日期对应一个标签
    with open(pripath, 'r') as rf:
        reader = csv.reader(rf)
        date = [row[0][5:] for row in reader]
        date.pop(0)

    # 取文件最后一列作为标签
    with open(labelpath, 'r') as rf1:
        reader = csv.reader(rf1)
        test_labels = [rs[10] for rs in reader]
        test_labels.pop(0)

    # 评论日期
    with open(textpath, 'r', encoding='utf-8')as csvfile:
        r1 = csv.reader(csvfile)
        news_test_date = [row[4][0:5] for row in r1]
        news_test_date.reverse()
        #print(len(news_test_date))

    # 根据时间设置训练集的标签
    a = 0
    y_test_data = []
    for B in news_test_date:
        try:
            if B == date[a]:
                y_test_data.append(test_labels[a])
            elif B == date[a + 1]:
                y_test_data.append(test_labels[a + 1])
                a += 1
            else:
                y_test_data.append(test_labels[a])
        except IndexError:
            break
    count = len(y_test_data) - 1
    for i, vec in enumerate(news_test_date, 0):
        if int(i) > count:
            y_test_data.append(test_labels[len(date) - 1])
    #print(len(y_test_data))
    y_test_data.pop(0)
    y_test_data = np.array(y_test_data)
    return y_test_data

#文件存储路径
x_path='data/all_news_train.csv'

y_pripath='data/sh.601628_train_data.csv'
y_textpath='data/newsVec_train.csv'
y_labelpath='data/sh.601628_train_data.csv'

#划分训练集和测试集
x_train=getX_train(x_path)
y_train=gety_train(y_pripath,y_textpath,y_labelpath)
print(x_train.shape,y_train.shape)
x_test=getX_test(r'data/all_news.csv')
y_test=gety_test(r'data/sh.601601_data.csv',r'data/dfcw.csv',r'data/sh.601601_data.csv')

TIME_STEPS=1
INPUT_DIM=50
output_dim = 1
batch_size = 60   #选择一组来更新权重
epochs = 10  #所有训练集都经过
seq_len = 1
hidden_size = 128

#创建张量
inputs = Input(shape=(TIME_STEPS, INPUT_DIM))
print('The shape of inputs:',inputs.shape)
#卷积过程
x = Conv1D(filters = 128, kernel_size = 1, activation = 'relu')(inputs)  #, padding = 'same'
print('The shape of x:',x.shape)
x = MaxPooling1D(pool_size =1)(x)
print('The shape of x:',x.shape)
x = Dropout(0.6)(x)
x=Conv1D(filters = 64, kernel_size =1, activation = 'relu')(x)
print('The shape of x:',x.shape)
x=Conv1D(filters = 32, kernel_size =1, activation = 'relu')(x)
#x = MaxPooling1D(pool_size = 3)(x)
print('The shape of x:',x.shape)
#平铺层，调整维度适应全链接层
x=Flatten(name='reshape_layer')(x)
output = Dense(1, activation='sigmoid')(x)
model = Model(inputs=inputs, outputs=output)
#打印模型概况
print(model.summary())
#均方误差，adam优化
model.compile(loss='mean_squared_error', optimizer='adam')
#拟合模型
model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size, shuffle=False)
#保存模型
save_dir = os.path.join(os.getcwd(), 'demo/model')
model_name = 'keras_CNN_trained_model.h5'
filepath="model_{epoch:02d}-{val_acc:.2f}.hdf5"
checkpoint = ModelCheckpoint(os.path.join(save_dir, filepath), monitor='val_acc',verbose=1,save_best_only=True)
if not os.path.isdir(save_dir):
    os.makedirs(save_dir)
model_path = os.path.join(save_dir, model_name)
model.save(model_path)
print('Saved trained model at %s ' % model_path)

model1=load_model(os.path.join(save_dir, model_name))
y_pred = model1.predict(x_test)
print(y_pred)

with open("demo/selfcoding_feature.csv", 'w', newline='') as t_file:
    csv_writer = csv.writer(t_file)
    for l in y_pred:
        csv_writer.writerow(l)