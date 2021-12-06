# -*- coding: utf-8 -*-
import  numpy as np
import  tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import  matplotlib.pyplot as plt
import  pandas as pd
from kt_package.Personal_module import plot2fig


data_train = pd.read_csv('fashion-mnist_train.csv',header=0)
data_test = pd.read_csv('fashion-mnist_test.csv',header=0)

labels = data_train['label'].values.reshape(1,60000)
labels_ = np.zeros((60000,10))
labels_[np.arange(60000),labels] = 1   # one-hot codec
#labels_ = labels_.transpose()
train = data_train.drop('label',axis=1)

labels_test = data_test['label'].values.reshape(1,10000)
labels_test_ = np.zeros((10000,10))
labels_test_[np.arange(10000),labels_test] = 1

# another method for one-hot codec
#labels_test = data_test['label']
#labels_test_  = tf.keras.utils.to_categorical(labels_test)

#labels_test_ = labels_test_.transpose()
test = data_test.drop('label',axis = 1)

print(labels_.shape)
print(labels_test_.shape)

# 规范化样本
train = np.array(train / 255.0)
test = np.array(test / 255.0)
labels_ = np.array(labels_)
labels_test = np.array(labels_test_)

# 构建网络
x = tf.placeholder(tf.float32,shape=[None,28*28])
x_image = tf.reshape(x,[-1,28,28,1])
y_true = tf.placeholder(tf.float32,shape=[None,10])
y_true_scalar = tf.argmax(y_true,axis = 1)

def new_conv_layer(input,num_input_channels,filter_size,num_filters):
    shape = [filter_size,filter_size,num_input_channels,num_filters]
    weights = tf.Variable(tf.truncated_normal(shape,stddev = 0.05))
    biases = tf.Variable(tf.constant(0.5,shape=[num_filters]))
    layer = tf.nn.conv2d(input=input,filter = weights,strides =[1,1,1,1],padding = 'SAME')
    layer += biases
    return layer,weights

def new_pool_layer(input):
    layer = tf.nn.max_pool(value=input,ksize=[1,2,2,1],strides = [1,2,2,1],padding='SAME')
    return layer

def new_relu_layer(input_layer):
    layer = tf.nn.relu(input_layer)
    return layer

def new_fc_layer(input,num_inputs,num_outputs):
    weights = tf.Variable(tf.truncated_normal([num_inputs,num_outputs],stddev = 0.05))
    biases = tf.Variable(tf.constant(0.05,shape=[num_outputs])) 
    layer = tf.matmul(input,weights) + biases
    return layer
    
layer_conv1,weights_conv1 = new_conv_layer(input = x_image,num_input_channels=1,filter_size=5,num_filters=6)
layer_pool1 = new_pool_layer(layer_conv1)
layer_relu1 = new_relu_layer(layer_pool1)
layer_conv2,weights_conv2 = new_conv_layer(layer_relu1, num_input_channels=6, filter_size=5, num_filters=16) 
layer_pool2 = new_pool_layer(layer_conv2)
layer_relu2 = new_relu_layer(layer_pool2)

# 全连接层
num_features = layer_relu2.get_shape()[1:4].num_elements() 
layer_flat = tf.reshape(layer_relu2,[-1,num_features])

# 创建最后几层
layer_fc1 = new_fc_layer(layer_flat,num_inputs=num_features,num_outputs=128)
layer_relu3 = new_relu_layer(layer_fc1)
layer_fc2 = new_fc_layer(input=layer_relu3,num_inputs=128,num_outputs=10) 

y_pred = tf.nn.softmax(layer_fc2)
y_pred_scalar = tf.argmax(y_pred,axis=1)

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=layer_fc2,labels=y_true)) 

optimizer = tf.train.AdamOptimizer(learning_rate=1e-4).minimize(cost)

correct_prediction = tf.equal(y_pred_scalar,y_true_scalar)
accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))

num_epochs = 200
batch_size = 100
def run_model(num_epochs,batch_size):    
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        train_accuracy_history = []
        test_accuracy_history = []
        
        for epoch in range(num_epochs):
            train_accuracy = 0
            for i in range(0,train.shape[0],batch_size):
                x_batch = train[i:i+batch_size,:]
                y_true_batch = labels_[i:i+batch_size,:]
                
                sess.run(optimizer,feed_dict={x:x_batch,y_true:y_true_batch})
                train_accuracy = sess.run(accuracy,feed_dict={x:x_batch,y_true:y_true_batch})
                
            #train_accuracy /= int(len(labels_)/batch_size)
            test_accuracy = sess.run(accuracy,feed_dict={x:test,y_true:labels_test})
            if epoch % 10 == 0:
                print('Reached epoch: %4d' %epoch,'  train acc:%.5f' %train_accuracy,'   test acc:%.5f' %test_accuracy)
                train_accuracy_history = np.append(train_accuracy_history,train_accuracy)
                test_accuracy_history = np.append(test_accuracy_history,test_accuracy)
            
            
    return sess,train_accuracy_history,test_accuracy_history

sess,train_accuracy,test_accuracy = run_model(num_epochs, batch_size)
plot2fig(train_accuracy,test_accuracy, title='Train & test Acccuracy', legend_str=['Train accuracy','Test accuracy'])
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    