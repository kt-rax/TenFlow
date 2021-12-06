# -*- coding: utf-8 -*-
import numpy as np
from sklearn.datasets import  fetch_openml
from sklearn.metrics import  confusion_matrix
import  tensorflow as tf
from kt_package.Personal_module import plot2fig

mnist = fetch_openml("mnist_784",version=1, cache=True)
X_input,y_input = mnist['data'].to_numpy(),mnist['target'].astype(np.float32).to_numpy() #.astype(np.int8)

# 重新分配标签：0-->0;1~9-->1
y_ = np.zeros_like(y_input)
y_[np.any([y_input==0],axis=0)] = 0
y_[np.any([y_input > 0],axis=0)] = 1

np.random.seed(42)
rnd = np.random.rand(len(y_)) < 0.8

x_train = X_input[rnd,:]
y_train = y_[rnd]
x_dev = X_input[~rnd,:]
y_dev = y_[~rnd]

# 标准化训练集
X_train_normalised = x_train/255.0

# 转置并准备张量
X_train_tr = X_train_normalised.transpose()
y_train_tr = y_train.reshape(1,y_train.shape[0])

# 重命名
xtrain = X_train_tr
ytrain = y_train_tr

# 构建网络

tf.reset_default_graph()

n_dim = 784

X = tf.placeholder(tf.float32,[n_dim,None])
Y = tf.placeholder(tf.float32,[1,None])
learning_rate = tf.placeholder(tf.float32,shape=())

W = tf.Variable(tf.zeros([1,n_dim]))
b = tf.Variable(tf.zeros(1))

init = tf.global_variables_initializer()
y_ = tf.sigmoid(tf.matmul(W,X)+b)

cost = -tf.reduce_mean(Y*tf.log(y_)+(1-Y)*tf.log(1-y_))
training_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)


correct_prediction = tf.equal(tf.greater(y_,0.5),tf.equal(Y,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))

def run_logistic_model(learning_r,training_epochs,train_obs,train_lables,debug=False):
    sess = tf.Session()
    sess.run(init)
    
    cost_history = []
    accuracy_history = []
    
    for epoch in range(training_epochs+1):
        sess.run(training_step,feed_dict={X:train_obs,Y:train_lables,learning_rate:learning_r})
        cost_ = sess.run(cost,feed_dict={X:train_obs,Y:train_lables,learning_rate:learning_r})
        accuracy_ = sess.run(accuracy,feed_dict={X:train_obs,Y:train_lables,learning_rate:learning_r})
        
        cost_history = np.append(cost_history,cost_)
        accuracy_history = np.append(accuracy_history,accuracy_)
        
        if (epoch % 10 == 0):
            print('Reached epoch',epoch,'cost J = ',str.format('{0:.6f}',cost_),'accuracy:',accuracy_)

    return sess,cost_history,accuracy_history

sess,cost_history,accuracy_history = run_logistic_model(learning_r=0.01,training_epochs=100,train_obs=xtrain,train_lables=ytrain,debug=False)

plot2fig(cost_history,accuracy_history,title='cost and accuracy',legend_str=['cost J','accuracy'])





