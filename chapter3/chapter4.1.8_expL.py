# -*- coding: utf-8 -*-
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior() 
#import tensorflow as tf
import numpy as np
import pandas as pd
from tensorflow.keras import backend as K
import matplotlib.pyplot as plt
import logging
from kt_package.Personal_module  import plot2fig 

logging.basicConfig(format='%(asctime)-15s '   )

# 定义一个创建层的函数 
def create_layer(X,n,activation):
    ndim = int(X.shape[0])
    stddev = 2/ np.sqrt(ndim)
    initialization = tf.truncated_normal((n,ndim),stddev = stddev)
    W = tf.Variable(initialization )
    b = tf.Variable(tf.zeros([n,1]))
    Z = tf.matmul(W,X)+b
    return activation(Z)

# 使用创建的层的函数来搭建网络
n_dim = 784
n1 = 600
n2 = 600
n3 = 600
n4 = 600
n_output = 10

X = tf.placeholder(tf.float32,[n_dim,None])
Y = tf.placeholder(tf.float32,[10,None])

learning_rate = tf.placeholder(tf.float32,shape=[])

# 学习率的衰减
initial_learning_rate = 0.1
decay_steps = 10000
decay_rate = 0.1
global_step = tf.Variable(0,trainable = False)
learning_rate_decay_2 = tf.train.inverse_time_decay(initial_learning_rate,global_step=global_step,decay_steps=decay_steps, decay_rate=decay_rate)

learning_rate_decay_3 = tf.train.polynomial_decay(initial_learning_rate,global_step,decay_steps, decay_rate)

learning_rate_decay_4 = tf.train.natural_exp_decay(initial_learning_rate,global_step,decay_steps, decay_rate)

boundaries = [10000,11000]
values = [1.0,0.5,0.1]

learning_rate_decay_5 = tf.train.piecewise_constant(global_step,boundaries,values)
'''
learning_rate_decay_6 = tf.train.exponential_decay(initial_learning_rate,global_step,
                                              decay_steps=decay_steps, decay_rate=decay_rate)

'''  

hidden1 = create_layer(X,n1,activation = tf.nn.relu)
hidden2 = create_layer(hidden1,n2,activation = tf.nn.relu)
hidden3 = create_layer(hidden2,n3,activation = tf.nn.relu)
hidden4 = create_layer(hidden3,n4,activation = tf.nn.relu)
outputs = create_layer(hidden4,n_output,activation = tf.identity)

y_ = tf.nn.softmax(outputs)

cost = -tf.reduce_mean(Y*tf.log(y_)+(1-Y)*tf.log(1-y_))
optimizer_1 = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

# 使用动态学习率衰减 
optimizer_2 = tf.train.GradientDescentOptimizer(learning_rate_decay_2).minimize(cost,global_step=global_step)

optimizer_3 = tf.train.GradientDescentOptimizer(learning_rate_decay_3).minimize(cost,global_step=global_step)

optimizer_4 = tf.train.GradientDescentOptimizer(learning_rate_decay_4).minimize(cost,global_step=global_step)

optimizer_5 = tf.train.GradientDescentOptimizer(learning_rate_decay_5).minimize(cost,global_step=global_step)
'''
optimizer_6 = tf.train.GradientDescentOptimizer(learning_rate_decay_6).minimize(cost,global_step=global_step)
'''

# 定义个model函数来运行模型
def model(process,optimizer,minibatch,training_epochs,features,classes,logging_step = 100,learning_r=0.001):
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    
    cost_history = []
    accuracy_history = []
    
    for epoch in range(training_epochs):
        for i in range(0,features.shape[1],minibatch):
            X_train_mini = features[:,i:i+minibatch]
            y_train_mini = classes[:,i:i+minibatch]
            
            sess.run(optimizer,feed_dict={X:X_train_mini,Y:y_train_mini,learning_rate:learning_r})
            
        cost_ = sess.run(cost,feed_dict={X:features,Y:classes,learning_rate:learning_r})
        
        cost_history = np.append(cost_history,cost_)
        
        if (epoch % logging_step == 0):
            print(process,' Reached epoch ',epoch,'cost J = ',cost_)
    
    # 增加精度打印
        correct_predictions = tf.equal(tf.argmax(y_,0),tf.argmax(Y,0))
        accuracy = tf.reduce_mean(tf.cast(correct_predictions,'float'))
        accuracy_ = accuracy.eval({X:features,Y:classes,learning_rate:learning_r},session=sess)
        accuracy_history = np.append(accuracy_history,accuracy_)
        print('Accuracy: ',accuracy.eval({X:features,Y:classes,learning_rate:learning_r},session=sess),' with ',epoch,' epochs')
    
    # plot
    #plt.figure()
    plt.figure
    plt.plot(cost_history)
    plt.title(process+' Datasets:'+str(features.shape[1])+' batch size:'+str(minibatch)+' Final Accuracy: '+str(accuracy.eval({X:features,Y:classes,learning_rate:learning_r},session=sess)))
    plt.xlabel('epoch')
    plt.ylabel('cost')
    plt.savefig(process+'.jpg')
    plt.close()
            
    return sess,cost_history,accuracy_history

# 寻找数据做好预处理，调用训练模型就可以啦

# 训练数据与标签 
train = pd.read_csv('fashion-mnist_train.csv')
labels = train['label'].values.reshape(1,60000)  # 根据lable键值提取标签 
train_ = train.drop('label',axis =1).transpose() # 在训练数据中不需要包含标签列，将其删除，其他保留，并转换为张量 
trainX = np.array(train_/255.0)  # 归一化

# 训练标签转化为softmax分类器的one-hot编码
labels_ = np.zeros((60000,10))
labels_[np.arange(60000),labels] = 1
labels_ = labels_.transpose()
labels_ = np.array(labels_)


# 测试数据与标签 
test = pd.read_csv('fashion-mnist_test.csv')
labels_dev = test['label'].values.reshape(1,10000) #根据label键值提取标签 
test_ = test.drop('label',axis =1).transpose()
testX = np.array(test_/255.0)

# 测试标签转为为softmax分类器的one-hot编码 
labels_dev_ = np.zeros((10000,10))               
labels_dev_[np.arange(10000),labels_dev] = 1
labels_dev_ = labels_dev_.transpose()
labels_dev_ = np.array(labels_dev_)

'''
correcttion_predictions = tf.equal(tf.argmax(y_,0),tf.argmax(Y,0))
accuracy = tf.reduce_mean(tf.cast(correcttion_predictions,'float'))
print('Accuracy:',accuracy.eval({X:,Y:,learning_rate:},session=sess))
'''

train_sess_1,train_history_1,accuracy_train_1 = model(r'Train',optimizer=optimizer_1,minibatch=100,training_epochs=150,features=trainX,classes=labels_,logging_step = 2,learning_r=0.01)
train_sess_2,train_history_2,accuracy_train_2 = model(r'Train',optimizer=optimizer_5,minibatch=100,training_epochs=150,features=trainX,classes=labels_,logging_step = 2,learning_r=0.01)

test_sess_1,test_history_1,accuracy_test_1 = model(r'Test',optimizer=optimizer_1,minibatch=100,training_epochs=150,features=testX,classes=labels_dev_,logging_step = 2,learning_r=0.01)
test_sess_2,test_history_2,accuracy_test_2 = model(r'Test',optimizer=optimizer_5,minibatch=100,training_epochs=150,features=testX,classes=labels_dev_,logging_step = 2,learning_r=0.01)

lr_strage = 'piecewise_constant'
'''
plt.figure
plt.plot(train_history_1)
plt.plot(train_history_2)
plt.legend(['constan_lr_train_loss',lr_strage+'_lr_trian_loss'])
plt.title('loss')
plt.xlabel('epoch')
plt.ylabel('cost')
plt.savefig(lr_strage+'_Train_loss'+'.jpg')
plt.close()
'''
plot2fig(train_history_1,train_history_2,title=lr_strage+'_Train_loss',legend_str=['constan_lr_train_loss',lr_strage+'_lr_trian_loss'])   

'''
plt.figure
plt.plot(accuracy_train_1)
plt.plot(accuracy_train_2)
plt.legend(['constan_train_accuracy',lr_strage+'_train_accuracy'])
plt.title('Accuracy')
plt.xlabel('epoch')
plt.ylabel('accuracy')
plt.savefig(lr_strage+'_Train_accuray'+'.jpg')
plt.close()
'''
plot2fig(accuracy_train_1,accuracy_train_2,title=lr_strage+'_Train_accuray',legend_str=['constan_train_accuracy',lr_strage+'_train_accuracy']) 
'''
plt.figure
plt.plot(train_history_1)
plt.plot(test_history_1)
plt.legend(['constan_lr_train_loss','constan_lr_test_loss'])
plt.title('loss')
plt.xlabel('epoch')
plt.ylabel('cost')
plt.savefig('const_lr_train_test_loss'+'.jpg')
plt.close()
'''
plot2fig(train_history_1,test_history_1,title='const_lr_train_test_loss',legend_str=['constan_lr_train_loss','constan_lr_test_loss']) 

'''
plt.figure
plt.plot(accuracy_train_1)
plt.plot(accuracy_test_1)
plt.legend(['constan_lr_train_accuracy','const_lr_test_accuracy'])
plt.title('Accuracy')
plt.xlabel('epoch')
plt.ylabel('accuracy')
plt.savefig('const_lr_train_test_accuracy'+'.jpg')
plt.close()
'''
plot2fig(accuracy_train_1,accuracy_test_1,title='const_lr_train_test_accuracy',legend_str=['constan_lr_train_accuracy','const_lr_test_accuracy'])             
'''            
plt.figure
plt.plot(train_history_2)
plt.plot(test_history_2)
plt.legend([lr_strage+'_train_loss',lr_strage+'_test_loss'])
plt.title('loss')
plt.xlabel('epoch')
plt.ylabel('cost')
plt.savefig(lr_strage+'_train_test_loss'+'.jpg')
plt.close()
'''
plot2fig(accuracy_train_1,accuracy_train_2,title=lr_strage+'_train_test_loss',legend_str=[lr_strage+'_train_loss',lr_strage+'_test_loss']) 
'''
plt.figure
plt.plot(accuracy_train_2)
plt.plot(accuracy_test_2)
plt.legend([lr_strage+'_train_accuracy',lr_strage+'_test_accuracy'])
plt.title('Accuracy')
plt.xlabel('epoch')
plt.ylabel('accuracy')
plt.savefig(lr_strage+'_train_test_accuracy'+'.jpg')
plt.close()
'''         
plot2fig(accuracy_train_1,accuracy_train_2,title=lr_strage+'_train_test_accuracy',legend_str=[lr_strage+'_train_accuracy',lr_strage+'_test_accuracy'])            
'''
plt.figure
plt.plot(test_history_1)
plt.plot(test_history_2)
plt.legend(['constant_test_loss',lr_strage+'_test_loss'])
plt.title('loss')
plt.xlabel('epoch')
plt.ylabel('cost')
plt.savefig(lr_strage+'_constant_test_loss'+'.jpg')
plt.close()
'''
plot2fig(accuracy_train_1,accuracy_train_2,title=lr_strage+'_constant_test_loss',legend_str=['constant_test_loss',lr_strage+'_test_loss']) 
'''
plt.figure
plt.plot(accuracy_test_1)
plt.plot(accuracy_test_2)
plt.legend(['constan_lr_test_accuracy',lr_strage+'_lr_test_accuracy'])
plt.title('Accuracy')
plt.xlabel('epoch')
plt.ylabel('accuracy')
plt.savefig(lr_strage+'_constant_test_accuracy'+'.jpg')
plt.close()            
'''                      
plot2fig(accuracy_test_1,accuracy_test_2,title=lr_strage+'_constant_test_accuracy',legend_str=['constan_lr_test_accuracy',lr_strage+'_lr_test_accuracy'])            

'''          
def plot2fig(x1,x2,title,legend_str):
    lr_strage = 'inverse_time_decay'
    
    plt.figure
    plt.plot(x1)
    plt.plot(x2)
    plt.legend(legend_str)
    plt.title(title)
    plt.xlabel('epoch')
    #plt.ylabel()
    plt.savefig(title+'.jpg')
    plt.close()
            
'''









            