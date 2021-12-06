# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt

def sigmoid(z):
    s = 1.0/(1.0+np.exp(-z))
    return s

def initialize(dim):
    w = np.zeros((dim,1))
    b = 0
    return w,b

def derivatives_calculation(w,b,X,Y):
    m = X.shape[1]
    z = np.dot(w.T,X) + b
    y_ = sigmoid(z)
    
    cost = -1.0/m*np.sum(Y*np.log(y_) + (1.0-Y)*np.log(1.0-y_))
    
    dw = 1.0/m*np.dot(X,(y_ - Y).T)
    db = 1.0/m*np.sum(y_ - Y)
    
    derivatives = {'dw':dw,'db':db}
    return derivatives,cost

def optimize(w,b,X,Y,num_iterations,learning_rate,print_cost=False):
    costs = [] 
    for i in range(num_iterations):
        derivatives, cost = derivatives_calculation(w, b, X, Y)
        dw = derivatives['dw']
        db = derivatives['db']
        
        w = w - learning_rate * dw
        b = b - learning_rate * db
        
        if i%100 == 0:
            costs.append(cost)
        
        if print_cost and i%100 == 0:
            print('cost (iteration % i) = %f' %(i,cost))
    
    derivatives = {'dw':dw,'db':db}
    params = {'w':w,'b':b}
    
    return params,derivatives,costs

def predict(w,b,X):
    m = X.shape[1]
    
    Y_prediction = np.zeros((1,m))
    w = w.reshape(X.shape[0],1)
    A = sigmoid(np.dot(w.T,X) + b)
    
    for i in range(A.shape[1]):
        if (A[:,i] > 0.5):
            Y_prediction[:,i] = 1
        elif (A[:,i] <= 0.5):
            Y_prediction[:,i] = 0
    return Y_prediction
  
def model (X_train,Y_train,X_test,Y_test,num_iterations= 1000,learning_rate =0.5,print_cost =False):

    w,b = initialize(X_train.shape[0])
    parameters, derivatives, costs = optimize(w,b,X_train,Y_train,num_iterations,learning_rate,print_cost)
    
    w = parameters['w']
    b = parameters['b']
    
    Y_prediction_test = predict(w, b, X_test)
    Y_prediction_train = predict(w, b, X_train)
    
    train_accuracy = 100 - np.mean(np.abs(Y_prediction_train - Y_train)*100.0)
    test_accuracy = 100 - np.mean(np.abs(Y_prediction_test - Y_test)*100.0)

    d = {'costs':costs,'Y_prediction_test':Y_prediction_test,'Y_prediction_train':Y_prediction_train,'w':w,'b':b,
         'learning_rate':learning_rate,'num_iterations':num_iterations}
    
    print('Accuracy Test:',test_accuracy)
    print('Accuracy Train',train_accuracy)
    
    return d
     
from sklearn.datasets import  fetch_openml
from sklearn.metrics import  confusion_matrix
import  tensorflow as tf
from kt_package.Personal_module import plot2fig

mnist = fetch_openml("mnist_784",version=1, cache=True)
X,y = mnist['data'].to_numpy(),mnist['target'].astype(np.float32).to_numpy() #.astype(np.int8)
x_12 = X[np.any([y==1,y==2],axis = 0)]
y_12 = y[np.any([y==1,y==2],axis = 0)]

shuffle_index = np.random.permutation(x_12.shape[0])
x_12_shuffled,y_12_shuffled = x_12[shuffle_index],y_12[shuffle_index]

train_proportion = 0.8
train_dev_cut = int(len(x_12)*train_proportion)

X_train,X_dev,Y_train,Y_dev = x_12_shuffled[:train_dev_cut,],x_12_shuffled[train_dev_cut:],y_12_shuffled[:train_dev_cut,],y_12_shuffled[train_dev_cut:]
                            
    
X_train_normalised = X_train / 255.0
X_dev_normalised = X_dev / 255.0

X_train_tr = X_train_normalised.transpose()
Y_train_tr = Y_train.reshape(1,Y_train.shape[0])

X_dev_tr = X_dev_normalised.transpose()
Y_dev_tr = Y_dev.reshape(1,Y_dev.shape[0])

Y_train_shifted = Y_train_tr - 1
Y_dev_shifted = Y_dev_tr -1

Xtrain,Ytrain,Xtest,Ytest = X_train_tr,Y_train_shifted,X_dev_tr,Y_dev_shifted

d = model(Xtrain,Ytrain,Xtest,Ytest,num_iterations=4000,learning_rate=0.05,print_cost=True)

    
  
    
  
    
  
    
  
    
  
    
  
    
  
    
  
    
  
    
  
    
  
    
  
    
  
    
  
    
  
    
  
    
  
    
  
    
  
    
  
    
  