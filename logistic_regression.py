import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

################################################################################

def sigmoid(z):
    return 1.0/(1+np.exp(-z))

###############################################################################

def normalize_features(X):
    # A is a matrix with each column representing one feature includeing x0
    m = X.shape[0]  # No. of training examples
    n = X.shape[1]  # No. of features
    normalized_X = np.zeros((m,n))
    
    for i in range(n):
        if np.amax(X[:,i])==0 and np.min(X[:,i]) == 0:
            pass
        else:
            normalized_X[:,i] = (X[:,i]-X[:,i].mean())/(X[:,i].max()-X[:,i].min())
        
    return normalized_X
    
################################################################################

def lrCostFunction(theta, X, y, reg_param):
    m, n = np.shape(X)

    J = (1.0/m)*sum(-y*np.log(sigmoid(X.dot(theta)))-(1-y)*np.log(1-sigmoid(X.dot(theta))))
    J = J + (reg_param/(2*m))*np.dot(theta[1:n].transpose(),theta[1:n])

    grad = (1.0/m)*(X.transpose().dot((sigmoid(X.dot(theta)))-y))
    rgrad = reg_param/m*theta
    rgrad[0] = 0
    grad = grad + rgrad

    return J, grad

################################################################################

def oneVsAll(X, y, num_labels, reg_param):
    m, n = np.shape(X)
    alltheta = np.zeros((num_labels,n+1))
    X = np.concatenate((np.ones((m,1)),X),axis=1)

    initialTheta = np.zeros((n+1,1))
    for i in range(num_labels):
        print('Started for:',i)
        minimizeResult = minimize(lrCostFunction,initialTheta,args=(X,(y == i).astype(int),reg_param),method='CG',jac=True)
        alltheta[i,:] = minimizeResult.x
        print('Done for:',i)
    return alltheta

################################################################################

def predictOneVsAll(all_theta, X):
    m = np.size(X,0)
    X = np.concatenate((np.ones((m,1)),X),axis=1)

    return np.argmax(X.dot(all_theta.transpose()),axis=1)

################################################################################

def main():
    X = np.loadtxt('ImagesPixelsData.txt',delimiter=',')
    y = np.loadtxt('Numbers.txt')
    y[0:500] = 0
    num_labels = 10
    reg_param = 0.1

    all_theta = oneVsAll(X, y, num_labels, reg_param)
    yp = predictOneVsAll(all_theta, X)

    np.savetxt('allTheta.txt',all_theta,delimiter=',')
    
    prediction_accuracy = np.mean((yp==y).astype(int)*100)
    print(prediction_accuracy)

################################################################################
if __name__ == "__main__":
    main()
