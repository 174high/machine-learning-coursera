# -*- coding: utf-8 -*

import numpy as np 
import matplotlib.pyplot as plt

#data = np.loadtxt('ex1data1.txt', delimiter=',', skiprows=1)
# skiprows=1 表示忽略第一行数据

data = np.loadtxt('ex1data1.txt', delimiter=',', skiprows=0) 

#print data 
#把数据全部打印出来

X= [x[0] for x in data]  


print(" len of X=%d "%len(X))

#print X 

y= [t[1] for t in data]

print(" len of y=%d"%len(y))

#print y 

plt.scatter(X, y)

plt.show()
