# -*- coding: utf-8 -*

import numpy as np 
import matplotlib.pyplot as plt



def computeCost(X,y,theta):

    m=len(y)

    y.resize((m,1))

    return sum(map(sum,np.power((np.dot(X.T,theta)-y),2)))/(2*m)


def gradientDescent(X,y,theta,alpha,num_iters):

    m=len(y)

    for i in range(num_iters):

	prediction=np.dot(X.T,theta)

	gredient=prediction-y 

	print "theta[0],thata[1] cost"

	print theta[0],theta[1],computeCost(X,y,theta)

	theta[0]=theta[0]-alpha/m*sum(map(sum,gredient))

	print "shape of X , ,prediction-y  "

	x=X[1]; 

	x.resize((m,1))

	print x.shape
	print gredient.shape

	gredient=np.dot(x.T,prediction-y)

	print "after calculate "

	print gredient.shape

	theta[1]=theta[1]-alpha/m*gredient

	print "theta[0],thata[1] cost"

	print theta[0],theta[1] ,computeCost(X,y,theta)



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

new_X=np.array((one,X))
y=np.array(y)

print new_X

print computeCost(new_X,y,theta)

gradientDescent(new_X,y,theta,alpha,iterations)









