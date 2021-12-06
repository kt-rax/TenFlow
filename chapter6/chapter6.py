# -*- coding: utf-8 -*-
import numpy as np
from sklearn.datasets import  fetch_openml

mnist = fetch_openml("mnist_784",version=1, cache=True)

total = 0

X,y = mnist['data'],mnist['target'].astype(np.int8)

for i in range(10):
    # 此处比较必须做数据类型转换，否则打印出全0次 
    print('digit',i,'appers',np.count_nonzero(y.astype(np.int) == i),'times')

for i in range(10):
    print('digit',i,'makes',np.around(np.count_nonzero(y==i)/70000.0*100.0,decimals=1),'% of the 70000 observations')
 
#
np.random.seed(1000)
rnd = np.random.rand(len(y))<0.8

train_y = y[rnd]
dev_y = y[~rnd]
    
for i in range(10):
    print('digit',i,'makes',np.around(np.count_nonzero(train_y == i)/56056.0*100,decimals=1),'% of the 56056 observations')
    
srt = np.zeros_like(y,dtype=bool)

np.random.seed(50)
srt[0:56000] = True

train_y = y[srt]
dev_y = y[~srt]

total = 0

for i in range(10):
    print('class',i,'makes',np.around(np.count_nonzero(train_y==i)/56000.0*100,decimals=1),'% of the 56000 observations')
    
    