# -*- coding: utf-8 -*-
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import numpy as np
import os
import matplotlib.pyplot as plt
import pandas as pd
from scipy.optimize import  curve_fit

files = os.listdir(r'./data/')

def get_T_O2(filename):
    T_ = float(filename[17:19])
    O2_ = float(filename[24:-4].replace('_','.'))
    return T_,O2_

def get_df(filenames):
    frame = pd.read_csv('./data/'+filenames,header = 10,sep = '\t')
    frame = frame.drop(frame.columns[6],axis = 1)
    
    frame.columns = ['f','ref_r','ref_phi','raw_r','raw_phi','sample_phi']
    
    return frame

# 遍历所有文件并创建列表
frame = pd.DataFrame()
df_list = []
T_list = []
O2_list = []

for file_ in files:
    df = get_df(file_)
    T_,O2_ = get_T_O2(file_)
    
    df_list.append(df)
    T_list.append(T_)
    O2_list.append(O2_)
    
# 频率转换
for df_ in df_list:
    df_['w'] = df_['f']*2*np.pi
    df_['tantheta'] = np.tan(df_['sample_phi']*np.pi/180.0)
    
T = 45
Tdf = pd.DataFrame(T_list,columns = ['T'])
Odf = pd.DataFrame(O2_list,columns = ['O2'])
filesdf = pd.DataFrame(files,columns = ['filename'])

files45 = filesdf[Tdf['T'] ==T ]
filesref = filesdf[(Tdf['T'] == T)&(Odf['O2'] == 0)]
fileref_idx = filesref.index[0]
O5 = Odf[Tdf['T'] == T]
dfref = df_list[fileref_idx]

from itertools import compress
A = Tdf['T'] == T
data = list(compress(df_list,A))

B = (Tdf['T'] == T)&(Odf['O2']==0)
dataref_ = list(compress(df_list,B))

# 遍历拟合
def fitfunc_2(x,f,KSV,KSV2):
    return (f/(1.0+KSV*x)+(1.0-f)/(1+KSV2*x))

f = []
KSV = []
KSV2 = []

for w_ in wred:
    O2x = []
    tantheta = []
    
#tantheta0 = float(dfref[dfref['w']==w_]['tantheta'])
tantheta0 = float(dataref_[0][dataref_[0]['w']==w_]['tantheta'])
# loop over the files
for idx,df_ in enumerate(data_train):
    O2xvalue = float(Odf.loc[idx])
    O2x.append(O2xvalue)
    
    tanthetavalue = float(df_[df_['w'] == w_]['tantheta'])
    tantheta.append(tanthetavalue)
    
popt,pcov = curve_fit(fitfunc_2,O2x,np.array(tantheta)/tantheta0,p0 = [0.4,0.06,0.003])

f.append(popt[0])
KSV.append(popt[1])
KSV2.append(popt[2])

# 首先只选择3000~100000之间的角频率对应的值
w_ = w[4:20]
f_ = f[4:20]
KSV_ = KSV[4:20]

# 插值
from scipy.interpolate import  interp1d

finter = interp1d(wred,f,kind = 'cubic')
KSVinter = interp1d(wred,KSV,kind='cubic')
KSV2inter = interp1d(wred,KSV2,kind ='cubic')

number_of_samples = 5000
number_of_x_points = len(w_)
np.random.seed(20)

O2_v = np.random.random_sample([number_of_samples])*100.0

# 需要的数学函数
def fitfunc2(x,O2,ffunc,KSVfunc,KSV2func):
    output = []
    for x_  in x:
        KSV_ = KSVfunc(x_)
        KSV2_ = KSV2func(x_)
        f_ = ffunc(x_)
        output_ = f_/(1.0+KSV_*O2)+(1.0-f_)/(1.0+KSV2_*O2)
        output.append(output_)
        
    return output

data = np.zeros((number_of_samples,number_of_x_points))
targets = np.reshape(O2_v,[number_of_samples,1])

for i in range(number_of_samples):
    data[i,:] = fitfunc2(w_,float(targets[i],finter,KSVinter,KSV2inter))
    
    
# 模型训练
tf.reset_default_graph()

n1 = 5
n2 = 5
n3 = 5
nx = number_of_x_points
n_dim = nx
n4 = 1

stddev_f = 2.0

tf.set_random_seed(5)

X = tf.placeholder(tf.float32,[n_dim,None])
Y = tf.palceholder(tf.float32,[10,None])
W1 = tf.Variable(tf.random_normal([n1,n_dim],stddev=stddev_f))
b1 = tf.Variable(tf.constant(0.0,shape=[n1,1]))
W2 = tf.Variable(tf.random_normal([n2,n_dim],stddev=stddev_f))
b2 = tf.Variable(tf.constant(0.0,shape=[n2,1]))
W3 = tf.Variable(tf.random_normal([n3,n2],stddev=stddev_f))
b3 = tf.Variable(tf.constant(0.0,shape=[n3,1]))
W4 = tf.Variable(tf.random_normal([n4,n3],stddev=stddev_f))
b4 = tf.Variable(tf.constant(0.0,shape=[n4,1]))

X = tf.placeholder(tf.float32,[nx,None])
Y = tf.placeholder(tf.float32,[1,None])

Z1 = tf.nn.sigmoid(tf.matmul(W1,X)+b1)
Z2 = tf.nn.sigmoid(tf.matmul(W2,Z1)+b2)
Z3 = tf.nn.sigmoid(tf.matmul(W3,Z2)+b3)
Z4 = tf.matmul(W4,Z3)+b4
y_ = Z2

y_ = tf.sigmoid(Z2)*100.0
cost = tf.reduce_mean(tf.square(y_-T))
learning_rate = 1e-3
train_step = tf.train.AdamOptimizer(learning_rate).minimize(cost)
init = tf.global_variables_initializer()

batch_size = 100
sess = tf.Session()
sess.run(init)
training_epochs = 25000
cost_history = np.empty(shape=[1],dtype=float)
train_x = np.transpose(data)
train_y = np.transpose(targets)
cost_histry = []

for epoch in range(training_epochs+1):
    
    for i in range(0,train_x.shape[0],batch_size):
        x_batch = train_x[i:i+batch_size,:]
        y_batch = train_y[i:i+batch_size,:]
        
        sess.run(train_step,feed_dict={X:x_batch,Y:y_batch})
        
    cost_ = sess.run(cost,feed_dict={X:train_x,Y:train_y})
    cost_history = np.append(cost_history,cost_)
    
    if(epoch % 1000 == 0):
        print('reached epoch',epoch,'costt J= ',cost_)
        






















































































