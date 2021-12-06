# -*- coding: utf-8 -*-
import  tensorflow as tf
'''
x1 = tf.constant(1)
x2 = tf.constant(2)
z = tf.add(x1,x2)

sess=tf.Session()
print(sess.run(z))
print(sess.run(x1))

x1 = tf.Variable(1)
x2 = tf.Variable(2)

z = tf.add(x1,x2)

sess = tf.Session()
# Tensorflow不会自动初始化变量，必须使用initializer使变量初始化，这样做比较麻烦

sess.run(x1.initializer)
sess.run(x2.initializer)

# 更好的方法是在计算图中添加一个节点，以便使用如下代码初始化在图中定义的所有变量 
init = tf.global_variables_initializer()
sess.run(init)
print(sess.run(z))


x1 = tf.placeholder(tf.float32,1)
x2 = tf.placeholder(tf.float32,1)
z = tf.add(x1,x2)

feed_dict = {x1:[1],x2:[2]}

sess = tf.Session()
print(sess.run(z,feed_dict))


x1 = tf.placeholder(tf.float32,[2])
x2 = tf.placeholder(tf.float32,[2])

z = tf.add(x1,x2)
feed_dict = {x1:[1,5],x2:[1,1]}

sess = tf.Session()
print(sess.run(z,feed_dict))

# 占位符定义时，必须始终将维度作为第二个输入参数传入
# 占位符的赋值可以通过一个包含所有占位符的名称作为键值的python字典来实现  
x1 = tf.placeholder(tf.float32,1)
w1 = tf.placeholder(tf.float32,1)
x2 = tf.placeholder(tf.float32,1)
w2 = tf.placeholder(tf.float32,1)

z1 = tf.multiply(x1,w1)
z2 = tf.multiply(x2,w2)
z3 = tf.add(z1,z2)

feed_dict = {x1:[1],w1:[2],x2:[3],w2:[4]}

sess = tf.Session()
print(sess.run(z3,feed_dict))


tf.reset_default_graph()

x1 = tf.constant(1)
x2 = tf.constant(2)
z = tf.add(x1,x2)
init = tf.global_variables_initializer()

sess = tf.Session()
sess.run(init)
#print(sess.run(z))
# sess.run()函数的参数就是要计算的节点的名称 
#print(sess.run([x1,x2,z]))
#print(sess.run([init,x1,x2,z]))
print(z.eval(session=sess))

tf.reset_default_graph()
c = tf.constant(5)
x = c + 1
y = x + 1
z = x + 2

sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)
print(sess.run(y))
print(sess.run(z))
#sess.close()

yy,zz = sess.run([y,z])

sess.close()


tf.reset_default_graph()

x = tf.constant(5)
y = x + 1
z = x + 2
#init = tf.global_variables_initializer()


with tf.Session() as sess:
    #sess.run(init)
    print(sess.run([y,z]))
    print(z.eval())


import random,time,timeit
from timeit import  Timer
import  logging
import numpy as np


FORMAT = '%(asctime)-15s %(clientip)s %(user)-8s %(message)s'   

    #运行5次测试模型的检测时间 
logging.basicConfig(format=FORMAT)
d = {'clientip': '192.168.0.1', 'user': 'fbloggs'}

# for循环与numpy计算时间的对比实验 

list1 = random.sample(range(1,10**8),10**7)
list2 = random.sample(range(1,10**8),10**7)

# 使用for循环来计算
logger = logging.getLogger('位置')
logger.warning('' ,extra=d)

ab = [list1[i]*list2[i] for i in range(len(list1))]
    
logger = logging.getLogger('位置')
logger.warning('' ,extra=d)

# 使用numpy 来计算 
list1_np = np.array(list1)
list2_np = np.array(list2)

logger = logging.getLogger('位置')
logger.warning('' ,extra=d)

out2 = np.multiply(list1_np,list2_np)

logger = logging.getLogger('位置')
logger.warning('' ,extra=d)


import  numpy as np
#import  hypothesis

m = 30
w0 = 2
w1 = 0.5
x = np.linspace(-1,1,m)
y = w0 + w1*x

def hypothesis(x,w0,w1):
    return w0+w1*x

Fun = np.average((y-hypothesis(x,w0,w1))**2,axis=2)/2

# chapter2.1

import matplotlib.pyplot as plt
import  tensorflow as tf
import  numpy as np
from sklearn.datasets import load_boston

boston = load_boston()
features = np.array(boston.data)
labels = np.array(boston.target)

print(boston['DESCR'])

n_training_samples = features.shape[0]
n_dim = features.shape[1]
print('The dataset has',n_training_samples,'training samples.')
print('The dataset has',n_dim,'features.')

def normalize(dataset):
    mu = np.mean(dataset,axis = 0)
    sigma = np.std(dataset,axis = 0)
    return (dataset - mu)/sigma

# 对特征进行归一化 
features_norm = normalize(features)

# 训练数据定义
train_x = np.transpose(features_norm)
train_y = np.transpose(labels)

print(train_x.shape)
print(train_y.shape)

# Tensorflow需要的维度表示应该是（1,506），因此必须进行重组
train_y = train_y.reshape(1,len(train_y))
print(train_y.shape)

tf.reset_default_graph()

# 计算阶段不发生更改的实体使用placeholder，但可能随没次运行而变化，包含输入数据集，学习率
X = tf.placeholder(tf.float32,[n_dim,None])
Y = tf.placeholder(tf.float32,[1,None])
learning_rate = tf.placeholder(tf.float32,shape=[])

# 计算过程中发生变化的实体定义成变量，像权重与偏执 
W = tf.Variable(tf.ones([n_dim,1]))
b = tf.Variable(tf.zeros(1))

# 全局变量一次性初始化      
init = tf.global_variables_initializer()

# 定义激活函数，使用的是恒等激活函数 
yy = tf.matmul(tf.transpose(W),X) + b

# 定义代价函数/损失函数              
cost = tf.reduce_mean(tf.square(yy - Y))

# 定义优化器
training_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)


# 定义一个实际学习的函数来封装功能  
def run_linear_model(learning_r,training_epochs,train_obs,train_labels,debug = False):
      sess = tf.Session()
      sess.run(init)
      
      cost_history = np.empty(shape = [0],dtype = float)
      
      for epoch in range(training_epochs+1):
          #注意使用feed_dict字典对所有的占位符placeholder进行赋值 
          sess.run(training_step,feed_dict={X:train_obs,Y:train_labels,learning_rate:learning_r})
          
          cost_ = sess.run(cost,feed_dict = {X:train_obs,Y:train_labels,learning_rate:learning_r})
          
          cost_history = np.append(cost_history,cost_)
          
          if(epoch % 1000 ==0) & debug:
              print('Reached epoch',epoch,'cost J = ',str.format(('{0:.6f}'),cost_))

      return sess,cost_history


# 调用训练函数进行实际的训练
sess,cost_history = run_linear_model(learning_r = 0.01, training_epochs = 10000, train_obs = train_x,
                                     train_labels = train_y, debug = True)

# plot 结果
plt.plot(cost_history)
plt.title('Cost History')
plt.xlabel('Test Step')
plt.ylabel('Cost History')
plt.legend(['cost history'],loc='upper right')
plt.savefig('cost_history')
plt.show()

'''
         
# chapter2.2

from sklearn.datasets import fetch_openml
import numpy as np
import matplotlib.pyplot as plt

mnist = fetch_openml('mnist_784', version=1, cache=True)
mnist.target = mnist.target.astype(np.int8)

# 加载数据集
#mnist = fetch_openml('MNIST original')
X,y = mnist['data'],mnist['target']   



for i in range(10):
    print('digit',i,'appears',np.count_nonzero(y == i),'times')

def plot_digit(some_digit):
    
    some_digit_image = some_digit.reshape(28,28)
    
    plt.imshow(some_digit_image,cmap = plt.cm.binary,interpolation = 'nearest')
    plt.axis('off')
    plt.show()

# 要实现的模型是二元分类的简单逻辑回归，因此数据集减少到两类 
X_train = X[np.any([y==1,y==2],axis = 0)]
y_train = y[np.any([y==1,y==2],axis = 0)]

# 必须对输入样本进行归一化（使用sigmoid激活函数时，不希望输入数据太大，因为有784个输入）
X_train_normalised = X_train/255.0

# 希望每列有一个输入样本，并且每行代表一个特征，因此必须重建张量
X_train_tr  = X_train_normalised.transpose()
y_train_tr = y_train.reshape(1,y_train.shape[0])

# 定义一个dim来表示特征的数量
n_dim = X_train_tr.shape[0]

# 导入的数据集中度标签将为1或2，但是我将使用的类的标签为0或1的假设来构建代价函数，因此必须重建调整y_train_tr
y_train_shift = y_train_tr - 1

# 
Xtrain = X_train_tr
ytrain = y_train_shift


tf.reset_default_graph()

X = tf.placeholder(tf.float32,[n_dim,None])
Y = tf.placeholder(tf.float32,[1,None])
learning_rate = tf.placeholder(tf.float32,shape=())

W = tf.Variable(tf.zeros([1,n_dim]))
b = tf.Variable(tf.zeros(1))

init = tf.global_variables_initializer()

# sigmoid激活函数
y_ = tf.sigmoid(tf.matmul(W,X)+b)
cost = -tf.reduce_mean(Y*tf.log(y_)+(1-Y)*tf.log(1-y_))
training_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

def run_logistic_model(learning_r,training_epochs,train_obs,train_labels,debug = False):
    sess = tf.Session()
    sess.run(init)
    
    cost_history = np.empty(shape = [0],dtype = float)
    
    for epochs in range(training_epochs+1):
        sess.run(training_step, feed_dict = {X:train_obs,Y:train_labels,learning_rate:learning_r})
        #sess.run(training_step,feed_dict={X:train_obs,Y:train_labels,learning_rate:learning_r}) 
        cost_ = sess.run(cost,feed_dict={X:train_obs,Y:train_labels,learning_rate:learning_r})
        
        cost_history = np.append(cost_history,cost_)
        
        if(epochs % 500 == 0) & debug:
            print('Reach epoch',epochs,'cost J = ',str.format('{0:.6f}',cost_))
    return sess,cost_history

sess,cost_history = run_logistic_model(learning_r=0.01,training_epochs=1000,train_obs=Xtrain,
                                       train_labels=ytrain,debug=True)

plt.plot(cost_history)
plt.xlabel('epochs')
plt.ylabel('cost')
plt.title('逻辑分类训练cost')
plt.savefig('chapter2.2.1.jpg')

        
def run_logistic_model_v2(learning_r,training_epochs,train_obs,train_labels,debug = False):
    sess = tf.Session()
    sess.run(init)
    
    cost_history = np.empty(shape =[0],dtype = float)
    
    for epoch in range(training_epochs):
        print('epoch:',epoch)
        print(sess.run(b,feed_dict={X:train_obs,Y:train_labels,learning_rate:learning_r}))
        
        sess.run(training_step,feed_dict={X:train_obs,Y:train_labels,learning_rate:learning_r})  
        print(sess.run(b,feed_dict={X:train_obs,Y:train_labels,learning_rate:learning_r}))
        
        cost_ = sess.run(cost,feed_dict={X:train_obs,Y:train_labels,learning_rate:learning_r})       
        cost_history = np.append(cost_history,cost_)
        
        if (epoch % 500 == 0) & debug:
            print('Reached epoch',epoch,'cost J = ',str.format('{0:.6f}',cost_))
    return sess,cost_history
            
          
sess,cost_history = run_logistic_model_v2(learning_r=0.01,training_epochs=2500,train_obs=Xtrain,
                                       train_labels=ytrain,debug=True)       

plt.plot(cost_history)
plt.xlabel('epochs')
plt.ylabel('cost')
plt.title('逻辑分类训练cost')
plt.savefig('chapter2.2.2.jpg')

 
 
sess,cost_history = run_logistic_model_v2(learning_r=0.005,training_epochs=2500,train_obs=Xtrain,
                                       train_labels=ytrain,debug=True)         
        
plt.plot(cost_history)
plt.xlabel('epochs')
plt.ylabel('cost')
plt.title('逻辑分类训练cost')
plt.savefig('chapter2.2.3.jpg')

        
correct_prediction1 = tf.equal(tf.greater(y_,0.5),tf.equal(Y,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction1,tf.float32))
print(sess.run(accuracy,feed_dict={X:Xtrain,Y:ytrain,learning_rate:0.05}))









































