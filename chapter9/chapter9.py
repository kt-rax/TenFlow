# -*- coding: utf-8 -*-
# 氧气浓度
import numpy as np
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import matplotlib.pyplot as plt 
from kt_package.Personal_module import plot2fig

def L(x,A):
    Y = A**2/(A**2+x**2)
    return Y

number_of_x_point = 100
min_x = 0.0
max_x = 5.0
x = np.arange(min_x,max_x,(max_x - min_x)/number_of_x_point)

# 生成样本
number_of_samples = 1000
np.random.seed(20)
A_v = np.random.normal(1.0,0.4,number_of_samples)
# A_v = np.random.random_sample([number_of_dev_samples])*3.0

for i in range(len(A_v)):
    if A_v[i] <= 0:
        A_v[i] = np.random.random_sample([1])
data = np.zeros((number_of_samples,number_of_x_point))
targets = np.reshape(A_v,[1000,1])
for i in range(number_of_samples):
    data[i,:] = L(x,A_v[i])

print(A_v.shape)

# 生成验证集
number_of_dve_samples = 1000

np.random.seed(42)
A_v_dev = np.random.normal(1.0,0.4,number_of_dve_samples)

for i in range(len(A_v_dev)):
    if A_v_dev[i] <= 0:
        A_v_dev[i] = np.random.random_sample([1])

data_dev = np.zeros((number_of_dve_samples,number_of_x_point))
targets_dev = np.reshape(A_v_dev,[1000,1]) 

for i in range(number_of_dve_samples):
    data_dev[i,:] = L(x,A_v_dev[i])

# 构建网络
tf.reset_default_graph()

n1 = 10
nx =  number_of_x_point
n2 = 1

W1 = tf.Variable(tf.random_normal([n1,nx]))/500.0
b1 = tf.Variable(tf.ones((n1,1)))/500.0
W2 = tf.Variable(tf.random_normal([n2,n1]))/500.0
b2 = tf.Variable(tf.ones((n2,1)))/500.0

X = tf.placeholder(tf.float32,[nx,None])  # inputs
Y = tf.placeholder(tf.float32,[1,None])   # labels

Z1 = tf.matmul(W1,X) + b1
A1 = tf.nn.sigmoid(Z1)
Z2 = tf.matmul(W2,A1) + b2
y_ = Z2
cost = tf.reduce_mean(tf.square(y_ - Y))
learning_rate = 0.1
training_step = tf.train.AdamOptimizer(learning_rate).minimize(cost)

init = tf.global_variables_initializer()

# 训练网络
sess = tf.Session()
sess.run(init)

training_epochs = 20000
cost_history = []
cost_dev_history = []

train_x = np.transpose(data)
train_y = np.transpose(targets)

for epoch in range(training_epochs+1):
    sess.run(training_step,feed_dict={X:train_x,Y:train_y})
    cost_ = sess.run(cost,feed_dict={X:train_x,Y:train_y})
    
    cost_history = np.append(cost_history,cost_)
 
    sess.run(training_step,feed_dict={X:data_dev.T,Y:targets_dev.T})
    cost_dev_ = sess.run(cost,feed_dict={X:data_dev.T,Y:targets_dev.T})
    cost_dev_history = np.append(cost_dev_history,cost_dev_)
    
    if(epoch % 1000 == 0):
        print('Reached epoch',epoch,'Train cost J= ',cost_,'Test cost',cost_dev_)
        
    
plot2fig(cost_history, cost_dev_history, title='Train and test cost with 20000 epochs', legend_str=['Train cost','Test cost'])    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
















































