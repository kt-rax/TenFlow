# -*- coding: utf-8 -*-
#import tensorflow as tf
#### could not run on the tensorflow 2.0 as the mnist data using method is not compareble

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

import tensorflow
import numpy as np
import  matplotlib.pyplot as plt
from kt_package.Personal_module import plot2fig
'''
#from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('C:/Users/KT/TenFlow/chapter4/',one_hot=True)

X_train = mnist.train.images.T
X_labels = mnist.train.labels.T

'''
# 
xx_train,xx_labels = tensorflow.keras.datasets.mnist.load_data() 


tf.reset_default_graph()

X = tf.placeholder(tf.float32,[784,None])
Y = tf.placeholder(tf.float32,[10,None])
learning_rate_ = tf.placeholder(tf.float32,shape=())
W = tf.Variable(tf.zeros([10,784]), dtype=tf.float32)
b = tf.Variable(tf.zeros([10,1]), dtype=tf.float32)

y_ = tf.nn.softmax(tf.matmul(W,X)+b)
cost = -tf.reduce_mean(Y*tf.log(y_)+(1-y_)*tf.log(1-y_))

grad_W,grad_b = tf.gradients(xs=[W,b],ys = cost)

new_W = W.assign(W - learning_rate_*grad_W)
new_b = b.assign(b - learning_rate_*grad_b)

#correct_prediction1= tf.equal(tf.greater(y_,0.5),tf.equal(Y,1))
correct_prediction1= tf.equal(tf.argmax(y_,0),tf.argmax(Y,0))
accuracy = tf.reduce_mean(tf.cast(correct_prediction1,'float32'))


# 学习率的衰减
initial_learning_rate = 0.1
decay_steps = 10000
decay_rate = 0.1
global_step = tf.Variable(0,trainable = False)
learning_rate_decay_2 = tf.train.inverse_time_decay(initial_learning_rate,global_step=global_step,decay_steps=decay_steps, decay_rate=decay_rate)

optimizer_1 = tf.train.RMSPropOptimizer(learning_rate=0.1,momentum=0.9).minimize(cost)
#optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate,beta1=0.9,beta2=0.999,epsilon=1e-8).minimize(cost)
optimizer_2 = tf.train.GradientDescentOptimizer(learning_rate_decay_2).minimize(cost,global_step=global_step)

def run_model_mb(optimizer,minbatch_size,training_epochs,features,classes,logging_step=100,learning_r=0.001):
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    
    total_batch = int(tensorflow.keras.datasets.mnist.train.num_examples/minbatch_size)
    
    cost_history = []
    accuracy_history = []
    
    for epoch in range(training_epochs+1):
        for i in range(total_batch):
 
            batch_xs,batch_ys = tensorflow.keras.datasets.mnist.train.next_batch(minbatch_size)
            batch_xs_t = batch_xs.T
            batch_ys_t = batch_ys.T
            sess.run(optimizer,feed_dict={X:batch_xs_t,Y:batch_ys_t,learning_rate_:learning_r})
            #_,_,cost_ = sess.run([new_W,new_b,cost],feed_dict={X:batch_xs_t,Y:batch_ys_t,learning_rate_:learning_r})
        cost_ = sess.run(cost,feed_dict={X:features,Y:classes})   
        accuracy_ = sess.run(accuracy,feed_dict={X:features,Y:classes})
        cost_history = np.append(cost_history,cost_)
        accuracy_history = np.append(accuracy_history,accuracy_)

        if (epoch % logging_step == 0):
            print("Reached epoch",epoch,'cost J= ',cost_)
            print("Accuracy:",accuracy_)
    
    return sess,cost_history,accuracy_history
            

sess,cost_history,accuracy_history = run_model_mb(optimizer_1,100,300,xx_train,xx_labels,logging_step=10,learning_r=0.1)
 
plot2fig(cost_history,accuracy_history,'cost and accuracy With epochs lr-'+initial_learning_rate,['cost','accuracy'])

sess,cost_history_,accuracy_history_ = run_model_mb(optimizer_2,100,300,xx_train,xx_labels,logging_step=10,learning_r=0.1)

plot2fig(cost_history_,accuracy_history_,'cost and accuracy With epochs lr-'+'inverse_time_decay',['cost','accuracy'])




        