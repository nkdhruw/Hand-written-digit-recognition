import numpy as np
import matplotlib.pyplot as plt
from utils import *

#-----------------------------------------------------------------------------#

def plotData(X,y):
      
    x1 = X[np.where(y==1),1][0]
    y1 = X[np.where(y==1),2][0] 
    x2 = X[np.where(y==0),1][0]
    y2 = X[np.where(y==0),2][0]

    plt.plot(x1,y1,'+',label='Positive')
    plt.plot(x2,y2,'o',label='Negative')
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.legend(loc=1)
    plt.show()

#-----------------------------------------------------------------------------#
    
def plotLineDivide(theta, X, y):

    x1 = X[np.where(y==1),1][0]
    y1 = X[np.where(y==1),2][0] 
    x2 = X[np.where(y==0),1][0]
    y2 = X[np.where(y==0),2][0]
    x3 = np.linspace((X[:,1]).min(),(X[:,1]).max(),101)
    y3 = -(theta[0]+theta[1]*x3)/theta[2]
    
    plt.plot(x1,y1,'+',label='Positive')
    plt.plot(x2,y2,'o',label='Negative')
    plt.plot(x3,y3)
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.legend(loc=1)
    plt.show()
    
#-----------------------------------------------------------------------------#

def plotDecisionBoundary(theta, X, y, order):
    x1 = np.linspace((X[:,1]).min(),(X[:,1]).max(),101)
    x2 = np.linspace((X[:,2]).min(),(X[:,2]).max(),101)
    X1, X2 = np.meshgrid(x1, x2)
    Z = np.zeros((101,101))
    xij = np.zeros((1,2))
    for i in range(101):
        for j in range(101):
            xij[0,0] = x1[i]
            xij[0,1] = x2[j]
            Z[i,j] = theta.dot(mapFeature(xij,order)[0])
            #print(Z[i,j])
    
    plt.figure()
    CS = plt.contour(X1, X2, Z,[0])
    plt.clabel(CS, inline=1, fontsize=10)
    
    x1 = X[np.where(y==1),1][0]
    y1 = X[np.where(y==1),2][0] 
    x2 = X[np.where(y==0),1][0]
    y2 = X[np.where(y==0),2][0]

    plt.plot(x1,y1,'+',label='Positive')
    plt.plot(x2,y2,'o',label='Negative')
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.legend(loc=1)
            