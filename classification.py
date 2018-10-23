import numpy as np
from scipy.optimize import minimize
from mlplot import *
from utils import *
#-----------------------------------------------------------------------------#
def sigmoid(z):
    return 1/(1+np.exp(-z))
#-----------------------------------------------------------------------------#
    
def costFunction(theta, X, y, regularization_param):
    m = np.size(X,0) # No. of training examples   
    h = sigmoid(theta.dot(X.transpose()))
    J = (1.0/m)*sum(-y*np.log(h)-(1-y)*np.log(1-h))+regularization_param/(2*m)*sum(theta*theta)  
    return J

#-----------------------------------------------------------------------------#

def costFunctionGradient(theta, X, y, regularization_param):
    m = np.size(X,0) # No. of training examples
    n = np.size(X,1) # No. of features including X0
    
    h = sigmoid(theta.dot(X.transpose()))
    grad = np.zeros((n))
    
    grad[0] = (1.0/m)*sum((h-y)*X[:,0])
    #grad[1:n] = (1.0/m)*np.dot((h-y),X[1:n,:]) + (regularization_param/m)*theta
    for j in range(1,n):
        grad[j] = (1.0/m)*sum((h-y)*X[:,j]) + regularization_param*theta[j]/m
    
    return grad   
#-----------------------------------------------------------------------------#

def optimizeTheta(initialTheta,regularization_param):
    minimizeResult = minimize(costFunction,initialTheta,args=(X,y,regularization_param),method='BFGS',jac=costFunctionGradient)
    return minimizeResult.x

#-----------------------------------------------------------------------------#
    
#def gradientDescent(X, y):
#    n = np.size(X,1)
#    theta = np.zeros((n))
#    alpha = 0.001
#    regularization_param = 0
#    tol = 1e-5
#    dJ = 100000
#    maxIter = 1000
#    itr = 0
#    J = 0
#    
#    while (dJ > tol):
#        if (itr==maxIter):
#            break
#        new_J, gradTheta = costFunction(theta, X, y, regularization_param)
#        theta = theta - alpha*gradTheta
#        dJ = abs(new_J-J)
#        J = new_J
#        print(itr)
#        itr += 1
#
#    if itr==maxIter:
#        print('Gradient descent did not converge')
#        exit()
#    else:
#        return theta

#-----------------------------------------------------------------------------#
        
data = np.loadtxt('ex2data2.txt',delimiter=',')
X = data[:,0:2]
y = data[:,2]
order = 4
X = mapFeature(X,order)
#print('Shape of X after feature addition:',np.shape(X))
initialTheta = np.zeros((np.size(X,1)))
regularization_param = 0
sol = optimizeTheta(initialTheta, regularization_param)

print('Solution:',sol)
#plotLineDivide(minimizeResult.x,X,y)
plotData(X,y)
plotDecisionBoundary(sol,X,y,order)
