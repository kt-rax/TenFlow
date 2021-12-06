# -*- coding: utf-8 -*-

import numpy as np
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import matplotlib.pyplot as plt
import pandas as pd
from random import shuffle

nn = 15
LL = 2**15
train_input = ['{0:015b}'.format(i) for i in range(LL)]
shuffle(train_input)

train_input = [map(int,i) for i in train_input]
temp = []
for i in train_input:
    temp_list = []
    for j in i:
        temp_list.append([j])
    temp.append(np.array(temp_list))
train_input = temp


train_output = []
for i in train_input:
    count = 0
    for j in i:
        if j[0] == 1:
            count += 1
    temp_list = ([0]*(nn+1))
    temp_list[count] = 1
    train_output.append(temp_list)

train_obs = LL-2000
dev_input = train_input[train_obs:]
dev_output = train_output[train_obs:]
train_input = train_input[:train_obs]
train_output = train_output[:train_obs]

tf.reset_default_graph()

data = tf.placeholder(tf.float32,[None,nn,1])
target = tf.placeholder(tf.float32,[None,(nn+1)])

num_hidden_el = 24
RNN_cell = tf.nn.rnn_cell.LSTMCell(num_hidden_el,state_is_tuple=True)
val,state = tf.nn.dynamic_rnn(RNN_cell,data,dtype=tf.float32)
val = tf.transpose(val,[1,0,2])
last = tf.gather(val,int(val.get_shape()[0]-1))
                 
W = tf.Variable(tf.truncated_normal([num_hidden_el,int(target.get_shape()[1])]))
b = tf.Variable(tf.constant(0.1,shape=[target.get_shape()[1]]))

prediction = tf.nn.softmax(tf.matmul(last,W) + b)
cross_entropy = -tf.reduce_sum(target*tf.log(tf.clip_by_value(prediction,1e-10,1.0)))
optimizer = tf.train.AdamOptimizer()
minimize = optimizer.minimize(cross_entropy)
errors = tf.not_equal(tf.argmax(target,1),tf.argmax(prediction,1))
error = tf.reduce_mean(tf.cast(errors,tf.float32)) 

init_op = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init_op)
mb_size = 1000
no_of_batches = int(len(train_input)/mb_size)
epoch = 50
for i in range(epoch):
    ptr = 0
    for j in range(no_of_batches):
        train,output = train_input[ptr:ptr+mb_size],train_output[ptr:ptr+mb_size]
        sess.run(minimize,{data:train,target:output})
    
    incorrect = sess.run(error,{data:dev_input,target:dev_output})
    print('Epoch {:2d} error {:3.1f}%'.format(i+1,100*incorrect)) 

   
    
                





































