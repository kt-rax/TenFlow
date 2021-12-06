# -*- coding: utf-8 -*-
#from sklearn.datasets import  fetch_mldata
import numpy as np
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import matplotlib.pyplot as plt

#mnist = fetch_mldata('MNIST original')
from sklearn.datasets import fetch_openml
#mnist = fetch_openml("mnist_784")
mnist = fetch_openml("mnist_784",version=1, cache=True)
#mnist.target = mnist.target

X,y = mnist['data'],mnist['target'].astype(np.int8)

for i in range(10):
    # 此处比较必须做数据类型转换，否则打印出全0次 
    print('digit',i,'appers',np.count_nonzero(y.astype(np.int) == i),'times')

# 只取1和2进行二元分类 
train_x = np.array(X[np.any([y==1,y==2],axis=0)])
train_y = np.array(y[np.any([y==1,y==2],axis=0)])
    
#train_x = np.transpose(X)
#train_y = np.transpose(y)


'''
print(train_x.shape)
(784, 70000)

print(train_y.shape)
(70000,)
'''
train_x = train_x.transpose()
train_y = train_y.reshape(1,len(train_y))
# train_y ===> (1,70000)

train_x = train_x/255.0
# 对输入样本进行归一化，必须的 

train_y = train_y - 1


n_dim = train_x.shape[0]
# 样本的特征数量

tf.reset_default_graph()

X = tf.placeholder(tf.float32,[n_dim,None])
Y = tf.placeholder(tf.float32,[1,None])
learning_rate = tf.placeholder(tf.float32,shape=())

W = tf.Variable(tf.zeros([1,n_dim]))
b = tf.Variable(tf.zeros(1))

init = tf.global_variables_initializer()

y_ = tf.sigmoid(tf.matmul(W,X)+b)    
cost = -tf.reduce_mean(Y*tf.log(y_)+(1-Y)*tf.log(1-y_))
training_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

def run_logistic_model(learning_r,training_epochs,train_obs,train_labels,debug=False):
    sess = tf.Session()
    sess.run(init)
    
    cost_history = np.empty(shape=[0],dtype=float)
    
    for epoch in range(training_epochs + 1):
        sess.run(training_step,feed_dict={X:train_obs,Y:train_labels,learning_rate:learning_r})
        cost_ = sess.run(cost,feed_dict={X:train_obs,Y:train_labels,learning_rate:learning_r})
        cost_history = np.append(cost_history,cost_)

        if (epoch % 250 == 0) & debug:
            print('Reached epoch',epoch,'cost J=' ,str.format('{0:.6f}',cost_))
    return sess,cost_history

sess,cost_history = run_logistic_model(learning_r= 0.005,training_epochs=2,train_obs=train_x,train_labels=train_y,debug=True)

plt.figure
plt.plot(cost_history)
plt.title('cost vs epochs')
plt.savefig('cost2.jpg')
#plt.imshow()






















