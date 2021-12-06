# -*- coding: utf-8 -*-
import pandas as pd
import tensorflow as tf
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from random import *

# 加载数据
data_train = pd.read_csv('fashion-mnist_train.csv',header=0)

labels = data_train['label'].values.reshape(1,60000)
train = data_train.drop('label',axis = 1).transpose()

# 归一化
train = np.array(train/255.0)

# 代价函数
#cost = -tf.reduce_mean(Y*tf.log(Y_)+(1-Y)*tf.log(1-Y_))

# one-hot编码
labels_ = np.zeros((60000,10))
labels_[np.arange(60000),labels] = 1
labels_ = labels_.transpose()
labels_ = np.array(labels_)

# Model
n_dim = 784
tf.reset_default_graph()

# Number of neurons in the layers
n1 = 5  # Number of neurons in layers1
n2 = 10 # Number of neurons in layers2
cost_history = np.empty(shape=[1],dtype = float)
learning_rate = tf.placeholder(tf.float32,shape=())

X = tf.placeholder(tf.float32,[n_dim,None])
Y = tf.placeholder(tf.float32,[10,None])
W1 = tf.Variable(tf.truncated_normal([n1,n_dim],stddev=0.1))
b1 = tf.Variable(tf.zeros([n1,1]))
W2 = tf.Variable(tf.truncated_normal([n2,n1],stddev=0.1))
b2 = tf.Variable(tf.zeros([n2,1]))

# Build the Network
Z1 = tf.nn.relu(tf.matmul(W1,X)+b1)
Z2 = tf.nn.relu(tf.matmul(W2,Z1)+b2)
y_ = tf.nn.softmax(Z2,0)

cost = -tf.reduce_mean(Y*tf.log(y_)+(1-Y)*tf.log(1-y_))
optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)

init = tf.global_variables_initializer()

sess = tf.Session()
sess.run(init)

training_epochs = 100

cost_history = []

'''
# =====>批量梯度下降，一个epoch周期更新一次权重与偏差 
for epoch in range(training_epochs+1):
    sess.run(optimizer,feed_dict={X:train,Y:labels_,learning_rate:0.01})
    
    cost_ = sess.run(cost,feed_dict={X:train,Y:labels_,learning_rate:0.01})
    
    cost_history = np.append(cost_history,cost_)
    
    if(epoch % 50 == 0):
        print('Reached epoch ',epoch,'cost J =' ,cost_)
        
correct_predictions = tf.equal(tf.argmax(y_,0),tf.argmax(Y,0))
accuracy = tf.reduce_mean(tf.cast(correct_predictions,'float'))
print('Accuracy:',accuracy.eval({X:train,Y:labels_,learning_rate:0.01},session=sess))
'''

features = train
classes = labels_

# ===>小批量梯度下降
for epoch in range(training_epochs+1):
    for i in range(0,features.shape[1],50):
        X_train_mini = features[:,i:i+50]
        y_train_mini = classes[:,i:i+50]

        sess.run(optimizer,feed_dict={X:X_train_mini,Y:y_train_mini,learning_rate:0.01})
        
        cost_ = sess.run(cost,feed_dict={X:features,Y:classes,learning_rate:0.01})
    
    cost_history = np.append(cost_history,cost_)
    if (epoch % 2 == 0):
        print('Reached epoch',epoch,'cost J= ',cost_)

# ===>SGD随机梯度下降
for epoch in range(training_epochs+1):
    for i in range(0,features.shape[1],1):
        X_train_mini = features[:,i:i+1]
        y_train_mini = classes[:,i:i+1]
        
        sess.run(optimizer,feed_dict={X:X_train_mini,Y:y_train_mini,learning_rate:0.01})
        cost_ = sess.run(cost,feed_dict={X:features,Y:classes,learning_rate:0.01})
    cost_history = np.append(cost_history,cost_)
        
    if(epoch % 2 == 0):
        print('Reached epoch',epoch,'cost J =',cost_)

#====定义训练函数，避免copy  
def model(minibatch_size,training_epochs,features,classes,loggeing_step=100,learning_r = 0.001):
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    
    cost_history = []
    
    for epoch in range(training_epochs+1):
        for i in range(0,features.shape[1],minibatch_size):
            X_train_mini,y_train_mini = features[:,i:i+minibatch_size],classes[:,i:i+minibatch_size]
            sess.run(optimizer,feed_dict={X:X_train_mini,Y:y_train_mini,learning_rate:learning_r})
            
        cost_ = sess.run(cost,feed_dict={X:features,Y:classes,learning_rate:learning_r})
        cost_history = np.append(cost_history,cost_)
    
    if(epoch % loggeing_step == 0):
        print('Reached epoch',epoch,'cost J =',cost_)
    
    return sess,cost_history



       

'''
# =========>测试 
data_dev = pd.read_csv('fashion-mnist_test.csv',header=0)
dev = data_dev.drop('label',axis = 1).transpose()
dev = np.array(dev/255.0)

label_dev = data_dev['label'].values.reshape(1,10000)
label_dev_ = np.zeros((10000,10))
label_dev_[np.arange(10000),label_dev] = 1
label_dev_ = label_dev_.transpose()

test_epochs = 1800

cost_history = []

for epoch in range(test_epochs+1):
    sess.run(optimizer,feed_dict={X:dev,Y:label_dev_,learning_rate:0.0005})
    
    cost_ = sess.run(cost,feed_dict={X:dev,Y:label_dev_,learning_rate:0.0005})
    
    cost_history = np.append(cost_history,cost_)
    
    if(epoch % 20 == 0 ):
        print('Reached epoch ',epoch,'cost J =' ,cost_)

correct_predictions = tf.equal(tf.argmax(y_,0),tf.argmax(Y,0))
accuracy = tf.reduce_mean(tf.cast(correct_predictions,'float32'))
print('Accuracy:',accuracy.eval({X:dev,Y:label_dev_,learning_rate:0.0005},session=sess))
print('correct_predictions:',correct_predictions.eval({X:dev,Y:label_dev_,learning_rate:0.001},session=sess))
'''


