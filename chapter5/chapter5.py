# -*- coding: utf-8 -*-
# 1.导库
import matplotlib.pyplot as plt
#import tensorflow as tf
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior() 
import numpy as np
from sklearn.datasets import  load_boston
import  sklearn.linear_model as sk
from kt_package.Personal_module import plot2fig

# 2.引数据集
boston = load_boston()
features = np.array(boston.data)
target = np.array(boston.target)

def normalize(datasets):
    mu = np.mean(datasets,axis=0)
    sigma = np.std(datasets,axis=0)
    return (datasets-mu)/sigma

# 数据归一化
features_norm = normalize(features)
np.random.seed(42)
rnd = np.random.rand(len(features_norm)) < 0.8

train_x = np.transpose(features_norm[rnd])
train_y = np.transpose(target[rnd])
dev_x = np.transpose(features_norm[~rnd])
dev_y = np.transpose(target[~rnd])

train_y = train_y.reshape(1,len(train_y))
dev_y = dev_y.reshape(1,len(dev_y))

def create_layer(X,n,activation):
    ndim = int(X.shape[0])
    stddev = 2.0 / np.sqrt(ndim)
    initialization = tf.truncated_normal_initializer((n,ndim),stddev=stddev)
    W = tf.Variable(initialization)
    b = tf.Variable(tf.zeros([n,1]))
    Z = tf.matmul(W,X)+b
    return activation(Z),W,b

tf.reset_default_graph

n_dim = 13
n1 = 20
n2 = 20
n3 = 20
n4 = 20
n_outputs = 1

tf.set_random_seed(5)

X = tf.placeholder(tf.float32,[n_dim,None])
Y = tf.placeholder(tf.float32,[1,None])

learning_rate = tf.placeholder(tf.float32,shape=())

hidden1,W1,b1 = create_layer(X,n1,activation=tf.nn.relu)
hidden2,W2,b2 = create_layer(hidden1,n2,activation=tf.nn.relu)
hidden3,W3,b3 = create_layer(hidden2,n3,activation=tf.nn.relu)
hidden4,W4,b4 = create_layer(hidden3,n4,activation=tf.nn.relu)
y_,W5,b5 = create_layer(hidden4,n_outputs,activation=tf.identity)

cost = tf.reduce_mean(tf.square(y_-Y))

optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate,beta1=0.9,beta2=0.999,eplison=1e-8).minimize(cost)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

cost_train_history = []
cost_dev_history = []

for epoch in range(10000+1):
    
    sess.run(optimizer,feed_dict={X:train_x,Y:train_y,learning_rate:0.001})
    cost_train_ = sess.run(cost,feed_dict={X:train_x,Y:train_y,learning_rate:0.001})
    cost_dev_ = sess.run(cost,feed_dict={X:dev_x,Y:dev_y,learning_rate:0.001})
    cost_train_history = np.append(cost_train_history,cost_train_)
    cost_dev_history = np.append(cost_dev_history,cost_dev_)
    
    if (epoch % 1000 == 0):
        print('Reached epoch',epoch,'cost J(train)=',cost_train_)
        print('Reached epoch',epoch,'cost J(test)=',cost_dev_)
    

plot2fig(cost_train_history, cost_dev_history, title='train & test cost ', legend_str=['cost_train','cost_test'])
    


























