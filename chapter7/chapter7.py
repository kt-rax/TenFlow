# -*- coding: utf-8 -*-

#### 黑盒优化： 网格搜索 
import numpy as np
from kt_package.Personal_module import plot1fig
import matplotlib.pyplot as plt

def f(x):
    tmp1 = -np.cos(x/4.0) - np.sin(x/4.0) - 2.5*np.cos(2.0*x/4.0) + 0.5*np.sin(2.0*x/4.0)
    tmp2 = -np.cos(x/3.0) - np.sin(x/3.0)- 2.5*np.cos(2.0*x/3.0) + 0.5*np.sin(2.0*x/3.0)
    return 10.0 + tmp1 + tmp2

xx = np.arange(1,100,1)

y = f(xx)

plot1fig(y,title='y',legend='y_fx')

gridsearch = np.arange(0,80,2)

x = 0
m = 0.0
### 搜索最大值 
'''
print(*np.ndenumerate(f(gridsearch)))
((0,), 3.0) ((1,), 6.206589580816521) ((2,), 11.358019773288868) ((3,), 12.239866271485816) ((4,), 9.333437735372039) ((5,), 8.054952195698078) ((6,), 10.577614346819963) ((7,), 13.310852130066078) ((8,), 12.83472684323257) ((9,), 10.61353467014556) ((10,), 9.74633290680248) ((11,), 9.815155720273879) ((12,), 8.347662685378273) ((13,), 6.152721378476711) ((14,), 7.097834885344442) ((15,), 11.759088518412003) ((16,), 14.959763351721813) ((17,), 12.261167305416432) ((18,), 6.65220864256902) ((19,), 5.230150651103624) ((20,), 9.812890697494321) ((21,), 14.557506643831507) ((22,), 13.921511415362984) ((23,), 9.69480327274155) ((24,), 7.615199996265159) ((25,), 9.236913954465422) ((26,), 10.810989538764773) ((27,), 9.724520232049878) ((28,), 8.099719879082043) ((29,), 8.601448826684019) ((30,), 9.948630057765623) ((31,), 9.523384525815121) ((32,), 8.500832706919093) ((33,), 10.46538502460164) ((34,), 14.971198691137749) ((35,), 16.465866295412162) ((36,), 11.612323208555342) ((37,), 4.831969336060336) ((38,), 3.336476390728894) ((39,), 7.905248497909919)
'''
for i,val in np.ndenumerate(f(gridsearch)):
    if( val > m ):
        m = val
        x = gridsearch[i]
        
print(x)
print(m)



plt.figure
plt.plot(y)
# plt 散点
plt.plot(gridsearch,f(gridsearch),'ro',color ='red')
plt.plot(x,m,'*',color='green')
# plt 散点拟合线 
#plt.plot(gridsearch,f(gridsearch),linestyle='dashdot',color ='red')
#plt.plot(gridsearch,f(gridsearch),linestyle='dashdot',color ='red',marker='o')
plt.title('gridsearch for f')
plt.legend(['f(x)','grid search f_','max'])
plt.savefig('gridsearch.jpg')
plt.close()


######## 网格搜索增加采样点数 :160
xlistg = []
flistg = []

for step in np.arange(1,20,0.5):        #
    gridsearch = np.arange(0,80,step)  #采样点数160 
    
    x = 0
    m = 0.0
    for i,val in np.ndenumerate(max(gridsearch)):
        if( val > m):
            m = val
            x = gridsearch[i]

    xlistg.append(x)     # 最大值的位置 
    flistg.append(m)     # 最大值 
    
plt.figure
plt.plot(flistg)
# plt 散点
#plt.plot(gridsearch,f(gridsearch),'ro',color ='red')
# plt 散点拟合线 
#plt.plot(gridsearch,f(gridsearch),linestyle='dashdot',color ='red')
#plt.plot(gridsearch,f(gridsearch),linestyle='dashdot',color ='red',marker='o')
plt.title('gridsearch for f')
#plt.legend(['f(x)','grid search f_'])
plt.savefig('gridsearch_160.jpg')
plt.close()

    
#### 随机搜索
import numpy as np

randomsearch = np.random.random([40])*80.0
np.random.seed(5)
randomsearch = np.random.random([10])*80

x = 0 
m = 0.0
for i,val in np.ndenumerate(f(randomsearch)):
    if (val > m):
        m = val
        x = randomsearch[i]
plt.figure
plt.plot(y)
# plt 散点
plt.plot(randomsearch,f(randomsearch),'ro',color ='red')
plt.plot(x,m,'*',color='green')
# plt 散点拟合线 
#plt.plot(gridsearch,f(gridsearch),linestyle='dashdot',color ='red')
#plt.plot(gridsearch,f(gridsearch),linestyle='dashdot',color ='red',marker='o')
plt.title('randsearch for f')
plt.legend(['f(x)','rand search f_','max'])
plt.savefig('randserach.jpg')
plt.close()

#####在x = 69.65858449419011的附近再采样10个点，从粗到细：现在大范围随机搜索，然后再根据结果再在小范围随机搜索
x = 69.65858449419011
randomsearch = x + (np.random.random([10])-0.5)*8

x = 0
m = 0.0
for i,val in np.ndenumerate(f(randomsearch)):
    if(val > m):
        m = val
        x = randomsearch[i]


plt.figure
plt.plot(y)
# plt 散点
plt.plot(randomsearch,f(randomsearch),'ro',color ='red')
plt.plot(x,m,'*',color='green')
# plt 散点拟合线 
#plt.plot(gridsearch,f(gridsearch),linestyle='dashdot',color ='red')
#plt.plot(gridsearch,f(gridsearch),linestyle='dashdot',color ='red',marker='o')
plt.title('randsearch for f')
plt.legend(['f(x)','rand search f_','max'])
plt.savefig('randserach_2.jpg')
plt.close()










































            
            
            
            
            
            
            
            
            
            
            
            
            
    
    
















































