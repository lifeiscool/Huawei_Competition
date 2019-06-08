from keras.layers import Input, Dense, LSTM
from keras.models import Model
import os
from keras.layers import *
from keras.models import *
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from numpy import *
set_printoptions(threshold=NaN)
from keras.callbacks import ModelCheckpoint
import pandas as pd
import csv
import matplotlib.pyplot as plt

#通过卷积自编码器获取价格特征

#参数设置
output_dim = 1
hidden_size = 128
batch_size = 60   #选择一组来更新权重
epochs = 500  #所有训练集都经过
seq_len = 1
TIME_STEPS = 1
INPUT_DIM = 9
lstm_units=10

def getdata(path):
    df=pd.read_csv(path)
    x=[0,1]
    df.drop(df.columns[x], axis=1, inplace=True)
    return df

#设置训练集
def get_train(df):
    data_train = df.iloc[:, :]
    print('data_train.shape', data_train.shape)
    scaler = MinMaxScaler()
    scaler.fit(data_train)
    data_x_train = scaler.transform(data_train)
    data_y_train=[[x[1]] for x in data_x_train]
    data_y_train.pop(0)
    data_y_train=np.array(data_y_train)
    print(data_y_train.shape)
    # 分割时间窗口
    X_train = np.array([data_x_train[i: i + seq_len, :] for i in range(data_x_train.shape[0] - seq_len)])
    y_train = np.array([data_y_train[i + seq_len, 0] for i in range(data_y_train.shape[0] - seq_len)])
    return  X_train,y_train

def get_test(df):
    data_test = df.iloc[:, :]
    print('data_test.shape', data_test.shape)
    scaler = MinMaxScaler()
    scaler.fit(data_test)
    data_x_test= scaler.transform(data_test)
    data_y_test = [[x[1]] for x in data_x_test]
    data_y_test.pop(0)
    data_y_test = np.array(data_y_test)
    print(data_y_test.shape)
    #分隔时间窗口
    X_test = np.array([data_x_test[i: i + seq_len, :] for i in range(data_x_test.shape[0] - seq_len)])
    y_test = np.array([data_y_test[i + seq_len, 0] for i in range(data_y_test.shape[0] - seq_len)])
    return X_test,  y_test

df1=getdata('data/sh.601628_train_data.csv')
df2=getdata('data/sh.601601_train_data.csv')
#分割数据集
X_train,y_train=get_train(df1)
X_test, y_test =get_test(df2)
print('XX',X_train.shape)
print('yy',y_train.shape)
save_dir = os.path.join(os.getcwd(), 'demo/model')
model_name = 'keras_selfcoding_trained_model.h5'

#设置输入形状
x = Input(shape=(TIME_STEPS,INPUT_DIM))
# Encoder
conv1_1 = Conv1D(64,  kernel_size = 3, activation='relu', padding='same',name='Conv_1')(x)
pool1 = MaxPooling1D(pool_size = 2, padding='same',name='Pool_1')(conv1_1)
conv1_2 = Conv1D(32, kernel_size = 3, activation='relu', padding='same',name='Conv_2')(pool1)
pool2 = MaxPooling1D(pool_size = 2, padding='same',name='Pool_2')(conv1_2)
conv1_3=Conv1D(1,kernel_size=3,activation='relu',padding='same',name='Conv_3')(pool2)
h= MaxPooling1D(pool_size = 1, padding='valid',name='Pool_3')(conv1_3)

# Decoder
conv2_1 = Conv1D(8,  kernel_size = 3, activation='relu', padding='same',name="Conv2_1")(h)
up1 = UpSampling1D(1,name="Samp_1")(conv2_1)
conv2_2 = Conv1D(16,  kernel_size = 3, activation='relu', padding='same',name="Conv2_2")(up1)
up2 = UpSampling1D(1,name="Samp_2")(conv2_2)
conv2_3=Conv1D(36,  kernel_size = 3, activation='relu', padding='same',name="Cone2_3")(up2)
up3=UpSampling1D(1,name="Samp_3")(conv2_3)
r = Conv1D(9, 4, activation='sigmoid', padding='same',name="Conv2_4")(up3)
#构建模型
autoencoder = Model(input=x, output=r)

#训练过程
# 编译
autoencoder.compile(optimizer='adam', loss='mse',metrics=['accuracy'])
#显示神经网络过程
print(autoencoder.summary())
#输入数据并训练
code=autoencoder.fit(X_train,X_train,batch_size=batch_size,epochs=epochs, shuffle=True)
#评价模型
score=autoencoder.evaluate(X_test,X_test,verbose=1)
print('loss:',score)

#保存模型
filepath="model_{epoch:02d}-{val_acc:.2f}.hdf5"
checkpoint = ModelCheckpoint(os.path.join(save_dir, filepath), monitor='val_acc',verbose=1,save_best_only=True)
if not os.path.isdir(save_dir):
    os.makedirs(save_dir)
model_path = os.path.join(save_dir, model_name)
autoencoder.save(model_path)
print('Saved trained model at %s ' % model_path)
model=load_model(os.path.join(save_dir, model_name))

#获取编码后的输出
coder=Model(inputs=model.input,outputs=model.get_layer('Pool_3').output)
feature=coder.predict(X_test)
print(feature)
print(len(X_test))

with open("demo/feature.csv", 'w', newline='') as t_file:
    csv_writer = csv.writer(t_file)
    for l in feature:
        csv_writer.writerow(l)


