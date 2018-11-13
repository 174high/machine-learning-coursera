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

#plt.show()

iterations = 1500;
alpha = 0.01;

theta=np.full([2,1], 0)
one=  np.full(len(y),1) 

print theta 
print one 

#new_X=[[0]*len(y)]*len(y) 

print "new x="
#print new_X

new_X=np.array((one,X))
y=np.array(y)

print new_X


prediction=np.dot(new_X.T,theta) 

print prediction 

print "shape of prediciton ="
print(prediction.shape)


print "shape of y ="

y.resize((97,1))

error=prediction-y 

print "error="

print error 

print "shape of error="
print(error.shape)










