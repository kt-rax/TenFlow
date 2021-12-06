# -*- coding: utf-8 -*-
### 贝叶斯优化
import numpy as np
from kt_package.Personal_module import plot1fig
import matplotlib.pyplot as plt

'''
def f(x):
    tmp1 = -np.cos(x/4.0) - np.sin(x/4.0) - 2.5*np.cos(2.0*x/4.0) + 0.5*np.sin(2.0*x/4.0)
    tmp2 = -np.cos(x/3.0) - np.sin(x/3.0)- 2.5*np.cos(2.0*x/3.0) + 0.5*np.sin(2.0*x/3.0) 
    return 10.0 + tmp1 + tmp2

xx = np.arange(1,100,1)

y = f(xx) 
'''      
def K(x,L,sigma = 1):
    return sigma**2*np.exp(-1.0/2.0/L**2*(x**2))

# 模拟未知函数的函数 
def f(x):
    return x**2 - x**3 + 10*x + 0.07*x**4

# 用5个点来构建f 
randompoints = np.random.random([5])*12.0
f_ = f(randompoints)

# 使用种子42产生随机数
np.random.seed(42)

#### 代码描述
xsampling = np.arange(0,14,0.2)

ybayes_ = []
sigmabayes_ = []

for x in xsampling:
    
    f1 = f(randompoints)
    sigma_ = np.std(f1)**2
    f_ = (f1 - np.average(f1))
    
    k = K(x - randompoints,2,sigma_)
    
    C = np.zeros([randompoints.shape[0],randompoints.shape[0]])
    Ctitle = np.zeros([randompoints.shape[0]+1,randompoints.shape[0]+1])
    for i1,x1 in np.ndenumerate(randompoints):
        for i2,x2 in np.ndenumerate(randompoints):
            C[i1,i2] = K(x1 - x2,2.0,sigma_)
            
    Ctitle[0,0] = K(0,2.0,sigma_)
    Ctitle[0,1:randompoints.shape[0]+1] = k.T
    Ctitle[1:,1:] = C
    Ctitle[1:randompoints.shape[0]+1,0] = k
    mu = np.dot(np.dot(np.transpose(k),np.linalg.inv(C)),f_)
    sigma2 = K(0,2.0,sigma_) - np.dot(np.dot(np.transpose(k),np.linalg.inv(C)),k)
    ybayes = np.asarray(ybayes_) + np.average(f1)
    sigmabayes_.append(np.abs(sigma2))

ybayes = np.asarray(ybayes_) + np.average(f1)
sigmabayes = np.asarray(sigmabayes_)


#### 替代函数评估函数
def get_surrogate(randompoints):
    ybayes_ = []
    sigmabayes = []
    for x in xsampling:
        
        f1 = f(randompoints)
        sigma_ = np.std(f1)**2
        f_ = (f1 - np.average(1))
        k = K(x - randompoints,2.0,sigma_)
        
        C = np.zeros([randompoints.shape[0],randompoints.shape[0]])
        Ctitle = np.zeros([randompoints.shape[0]+1,randompoints.shape[0]+1])
        for i1,x1 in np.ndenumerate(randompoints):
            for i2,x2 in np.ndenumerate(randompoints):
                C[i1,i2] = K(x1 - x2,2.0,sigma_)
        
        Ctitle[0,0] = K(0,2.0)
        Ctitle[0,1:randompoints.shape[0]+1] = k.T
        Ctitle[1:,1:] = C
        Ctitle[1:randompoints.shape[0]+1,0] = k
        
        mu = np.dot(np.dot(np.transpose(k),np.linalg.inv(C)),f_)
        sigma2 = K(0,2.0,sigma_) - np.dot(np.dot(np.transpose(k),np.linalg.inv(C)),k)
        ybayes_.append(mu)
        sigmabayes_.append(np.abs(sigma2))
        
    ybayes = np.asarray(ybayes_) + np.average(f1)
    sigmabayes = np.asarray(sigmabayes_)
    
    return ybayes,sigmabayes

plt.figure
plt.plot(f(randompoints),color='red')
plt.title('bayes opt')
plt.legend(['bayes'])
plt.savefig('bayes.jpg')
plt.close()


#### 评估新的点
def get_new_point(ybayes,sigmabayes,eta):
    idxmax = np.argmax(np.average(ybayes) + eta*np.sqrt(sigmabayes))
    newpoint = xsampling[idxmax]
    return newpoint

xmax = 40.0
numpoints = 6
xsampling = np.arange(0,xmax,0.2)
eta = 1.0

np.random.seed(8)
randompoints1 = np.random.random([numpoints])*xmax

ybayes1,sigmabayes1 = get_surrogate(randompoints1)

newpoint = get_new_point(ybayes1,sigmabayes1,eta)
randompoints2 = np.append(randompoints1,newpoint)
ybayes2,sigmabayes2 = get_surrogate(randompoints2)


#### chapter7.9 对数尺度采样
r = -np.arange(0,1,0.001)*4.0

point2 = 10**4

#print(np.sum(alpha<=1e-3)&(alpha>1e-4))
