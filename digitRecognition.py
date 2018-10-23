import numpy as np
import matplotlib.pyplot as plt
import cv2
from sklearn.datasets import fetch_mldata
from logistic_regression import *
from matplotlib_event_handling import *

        
##############################################################################

class MyPencil(Pencil):
    def __init__(self, fig, ax):
        Pencil.__init__(self, fig, ax)
        self.count = 1

    def onMove(self, event):
        if self.pressed is True:
            self.x.append(event.xdata)
            self.y.append(event.ydata)
            self.line.set_data(self.x,self.y)
            self.line.figure.canvas.draw()
            
    def onRelease(self, event):
        self.pressed = False
        self.fig.savefig('number.png')
        self.x = []
        self.y = []
        self.count += 1
        plotPredictedNumber()

        
##############################################################################

def normalizeImage(image):
    
    image = scaleImagePixelValues(image)
    image = removeWhiteSpaces(image)
    image = resizeImage(image)
    npixels_height, npixels_width = image.shape
    newImage = np.zeros((28,28),dtype='int8')

    hor_padding = 28-npixels_width
    lhp = hor_padding//2
    rhp = hor_padding - lhp
    
    ver_padding = 28-npixels_height
    tvp = ver_padding//2
    bvp = ver_padding - tvp

    newImage[tvp:tvp+npixels_height,lhp:lhp+npixels_width] = image

    return newImage

##############################################################################

def scaleImagePixelValues(image):
    image = image/(np.amax(image)-np.amin(image))
    return image

##############################################################################

def resizeImage(image):
    nrows, ncols = image.shape
    if nrows > ncols:
        image = cv2.resize(image, (round(20*ncols/nrows),20), interpolation=cv2.INTER_LINEAR)
    else:
        image = cv2.resize(image, (20,round(20*nrows/ncols)), interpolation=cv2.INTER_LINEAR)
    return image

##############################################################################

def removeWhiteSpaces(image):
    image[np.where(image>=0.5)]= 1
    image[np.where(image<0.5)] = 0

    corners_index = [0, 0, 0, 0] # left, right, top, bottom
    corners_found = [False, False, False, False]
    #npixels_width, npixels_height = image.shape
    npixels_height, npixels_width = image.shape

    for i in range(npixels_width):
        if not corners_found[0]:
            if image.sum(axis=0)[i]!=0:
                corners_index[0] = i
                corners_found[0] = True
        if not corners_found[1]:
            if image.sum(axis=0)[npixels_width-1-i]!=0:
                corners_index[1] = npixels_width-1-i
                corners_found[1] = True

    for i in range(npixels_height):
        if not corners_found[2]:
            if image.sum(axis=1)[i]!=0:
                corners_index[2] = i
                corners_found[2] = True
        if not corners_found[3]:
            if image.sum(axis=1)[npixels_height-1-i]!=0:
                corners_index[3] = npixels_height-1-i
                corners_found[3] = True

    l,r,t,b = corners_index
    image = image[t:b+1, l:r+1]
    return image

##############################################################################

def cleanse_data(X):
    m, n = X.shape
    X[np.where(X<=127)]= 0
    X[np.where(X>127)] = 1

    corners_index = [0, 0, 0, 0] # left, top, right, bottom
    corners_found = [False, False, False, False]
    for i in range(m):
        x = X[i,:].reshape((28,28))
        

##############################################################################

def fetch_data():
##    X = np.loadtxt('ImagesPixelsData.txt', delimiter=',')
##    y = np.loadtxt('Numbers.txt')
    mnist = fetch_mldata('MNIST original')
    X = mnist.data
    y = mnist.target
    return X, y

##############################################################################

def plotTrainingNumbers(X, nrows, ncols):
    m = X.shape[0]
    randNums = (m*np.random.rand(nrows, ncols)).astype(int)
    plt.subplots(nrows,ncols)
    idx = 1
    for row in range(nrows):
        for col in range(ncols):
            x = X[randNums[row,col],:]
            print('Original image:',x.reshape((28,28)).shape)
            image = normalizeImage(x.reshape((28,28)))
            plt.subplot(nrows,ncols,idx)
            idx += 1
            plt.imshow(image)
    plt.show()
    
##############################################################################

def train():
    X, y = fetch_data()
    m = X.shape[0]
    normX = X
    for i in range(m):
        x = X[i,:].reshape((28,28))
        normX[i,:] = normalizeImage(x).flatten()

    num_labels = 10
    reg_param = 0.1
    all_theta = oneVsAll(X, y, num_labels, reg_param)
    yp = predictOneVsAll(all_theta, X)
    np.savetxt('allTheta_MNIST.txt',all_theta,delimiter=',')
    
    prediction_accuracy = np.mean((yp==y).astype(int)*100)
    print(prediction_accuracy)
    
##############################################################################

def predictNumber():
    drawn_image = 255-cv2.imread('number.png',cv2.IMREAD_GRAYSCALE)
    normalized_image = normalizeImage(drawn_image)
  
    alltheta = np.loadtxt('allTheta_MNIST.txt',delimiter=',')
    predictedNumber = predictOneVsAll(alltheta, [normalized_image.flatten()])    

    return drawn_image, predictedNumber

##############################################################################

def plotPredictedNumber():
    drawn_image, predicted_num = predictNumber()
    plt.subplot(121), plt.imshow(drawn_image)
    plt.subplot(122), plt.text(0.3,0.3,str(predicted_num[0]),size=100)
    plt.draw()
    plt.show()
    
##############################################################################

def main():
    fig = plt.figure(frameon=False)
    fig.set_size_inches(3,3)
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)
    pencil = MyPencil(fig, ax)
    pencil.connect()
    plt.show() 
       
##############################################################################

if __name__ =="__main__":
    main()
        

    
