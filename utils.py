import numpy as np
#-----------------------------------------------------------------------------#
def mapFeature(X, order):
# X: two-dimensional array of two features, X1 and X2
    N = int((order+1)*(order+2)/2)
    m = np.size(X,0)
    Z = np.zeros((m,N))
    
    x1 = X[:,0]
    x2 = X[:,1]
    for j in range(order+1):
        ff_num = int(j*(j+1)/2)
        powers = []
        for k in range(j+1):
            feature_num = ff_num + k 
            #print('feature_num:',feature_num)
            Z[:,feature_num] = (x1**k)*(x2**(j-k))
            powers.append([k,j-k])
        #print(powers)
       
    #print(np.shape(Z))
    return Z

#-----------------------------------------------------------------------------#

if __name__ == "__main__":
    x1 = np.linspace(0,1,11)
    x2 = 2*x1
    X = np.zeros((11,2))
    X[:,0] = x1
    X[:,1] = x2
    X = mapFeature(X,4)