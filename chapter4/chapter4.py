# -*- coding: utf-8 -*-
import  tensorflow as tf

# 动态学习率衰减 
tf.train.exponential_decay()  # 指数衰减
tf.train.inverse_time_decay() # 逆时衰减
tf.train.natural_exp_decay()  # 自然指数衰减
tf.train.natural_decay()      
tf.train.piecewise_constant() # 步长衰减
tf.train.polynomial_decay()   # 多项式衰减 

initial_learning_rate = 0.1
decay_steps = 1000
decay_rate = 0.1
global_step = tf.Variable(0,trainable=False)
learning_rate_decay = tf.train.inverse_time_decay(initial_learning_rate,global_step,decay_steps, decay_rate)
#decayed_learning_rate = learning_rate/(1+decay_rate*global_step/decay_step)
optimizer = tf.train.GradientDescentOptimizer(learning_rate_decay).minimize(cost,global_step=global_step)
boundaries = [100000,110000]
values = [1.0,0.5,0.1]
global_step = tf.Variable(0,trainable=False)
boundaries = [100000,110000]
values = [1.0,0.5,0.1]
learning_rate = tf.train.piecewise_constant(global_step,boundaries, values)

