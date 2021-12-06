# -*- coding: utf-8 -*-

import matplotlib.pylab as plt

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
