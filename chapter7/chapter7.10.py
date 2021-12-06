# -*- coding: utf-8 -*-
#### zalando 超参调优

# 1.导库
import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from random import *

# 2.csv格式读取
data_train = pd.read_csv('fashion-mnist_train.csv',header = 0)
data_test = pd.read_csv('fashion-mnist_test.csv',header = 0)
print(data_train.shape)

# 3.导数据
labels = data_train['label'].values.reshape(1,60000)
labels_ = np.zeros((60000,10))
labels_[np.arange(60000),labels] = 1
labels_ = labels_.transpose()
train = data_train.drop('label',axis=1).transpose()
# 检测维度
print(labels_.shape,train.shape)

# 4.预处理
labels_test = data_test['label'].values.reshape(1,10000)
labels_test_ = np.zeros((10000,10))
labels_test_[np.arange(10000),labels_test] = 1
lables_test_ = labels_test_.transpose()
dev = data_test.drop('label',axis=1).transpose()

# 5.标准归一化特征
train = np.array(train / 255.0)
dev = np.array(dev / 255.0)
labels_ = np.array(labels_ )
labels_test_ = np.array(labels_test_)

def build_model(number_neurous):
    n_dim = 784
    tf.reset_default_graph()
    
    # 
    n1 = number_neurous
    n2 = 10
    
    cost_history = np.empty(shape=[1],dtype=float)
    learning_rate = tf.placeholder(tf.float32,shape=())
    
    X = tf.placeholder(tf.float32,[n_dim,None])
    Y = tf.placeholder(tf.float32,[10,None])
    W1 = tf.Variable(tf.truncated_normal_initializer([n1,n_dim],stddev=.1))
    b1 = tf.Variable(tf.constant(0.1,shape=[n1,1]))
    W2 = tf.Variable(tf.truncated_normal_initializer([n2,n1],stddev=.1))
    b2 = tf.Variable(tf.constant(0.1,shape=[n2,1]))
    
    #
    Z1 = tf.nn.rule(tf.matmul(W1,X)+b1) 
    Z2 = tf.matmul(W2,Z1) + b2
    y_ = tf.nn.softmax(Z2,0)
    
    cost = -tf.reduce_mean(Y*tf.log(y_)*tf.log(1-y_))
    optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

    init = tf.global_variables_initializer() 
    
    return optimizer,cost,y_,X,Y,learning_rate

# 定义训练模型函数
def model(minibatch_size,training_epoch,featrues,classes,logging_step=100,learning_r=0.001,number_neurous=15):
    opt,c,y_,X,Y,learning_rate = build_model(number_neurous)
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    
    cost_history = []
    
    for epoch in range(training_epoch+1):
        for i in range(0,featrues.shape[1],minibatch_size):
            x_train_mini = featrues[:,i:i+minibatch_size]
            y_train_mini = classes[:,i:i+minibatch_size]
            
            sess.run(opt,feed_dict={X:x_train_mini,Y:y_train_mini,learning_rate:learning_r})
            
        cost_ = sess.run(cost,feed_dict={X:featrues,Y:classes,learning_rate:learning_r})
        cost_history = np.append(cost_history,cost_)
        if(epoch % logging_step == 0):
            print('Reached epoch',epoch,'cost J=',cost_)
            
    correct_predictiions = tf.equal(tf.argmax(y_,0),tf.argmax(Y,0))
    accuracy = tf.reduce_mean(tf.cast(correct_predictiions,'float'))
    accuracy_train = accuracy.eval({X:train,Y:labels,learning_rate:learning_r},session = sess)
    accuracy_dev = accuracy.eval({X:dev,Y:dev,learning_rate:learning_r},session = sess)
    
    return accuracy_train,accuracy_dev,sess,cost_history
    
    
acc_train,acc_test,sess,cost_history = model(minibatch_size=50,training_epoch=100,featrues=train,classes=labels,logging_step=10,learning_r=0.001,number_neurous=15)    
    
        
## 网格搜索
nn = [1,5,10,15,25,30,50,150,300,1000,3000]
for nn_ in nn:
    acc_train,acc_test,sess,cost_history = model(minibatch_size=50,training_epoch=50,featrues=train,classes='labels',
                                                 logging_step=50,learning_r=0.001,number_neurous=nn_)

    print('Number of neurons:',nn_,'Acc.Train:',acc_train,'Acc.Test',acc_test)


###
'''
神经元数量：35~60
学习率：对数尺度1e-1 ~ 1e-3
小批量大小：20~80
周期数：40~100
'''
neurons_ = np.random.randint(low=35,high=60,size=(10))
r = -np.random.random([10])*3.0-1
learning_ = 10**r
mb_size_ = np.random.randint(low=20,high=80,size=10)
epochs_ = np.random.randint(low=40,high=100,size=(10))



for i in range(len(neurons_)):
    acc_train,acc_test,sess,cost_history = model_layers(minibatch_size = mb_size_,training_epoch=epochs_[i],
                                                        featrues=train,classes=labels_,logging_step=50,
                                                        learning_r = learning_[i],number_neurous=neurons_[i],debug=False)
    print('Epochs:',epochs_[i],'Number of neurons:',neurons_[i],'learning-rate：',learning_[i],'mb size:',mb_size_[i],
          'Acc.train:',acc_train,'Acc.test:',acc_test)
    























 
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
