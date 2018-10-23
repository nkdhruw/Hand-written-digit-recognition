import numpy as np
import matplotlib.pyplot as plt
from digitRecognition import *
from matplotlib.widgets import Button
from matplotlib.gridspec import GridSpec

################################################################################

class ButtonClickProcessor(object):
    def __init__(self, ax1, ax2, ax3):
        self.button = Button(ax1, 'Clear')
        self.ax2 = ax2
        self.ax3 = ax3
        self.button.on_clicked(self.clear)

    def clear(self, event):
        self.ax2.clear()
        self.ax3.clear()
        
################################################################################

class DigitRecognizer:
    def __init__(self):
        self.fig = plt.figure(figsize=(8,4), frameon=False)
        self.gs = GridSpec(5,6)
        self.ax1 = plt.subplot(self.gs[0,:])
        self.ax2 = plt.subplot(self.gs[0:,0:3])
        self.ax3 = plt.subplot(self.gs[0:,3:6])
        self.ax1.set_xticks([])
        self.ax1.set_yticks([])
        self.ax2.set_xticks([])
        self.ax2.set_yticks([])
        self.ax3.set_xticks([])
        self.ax3.set_yticks([])
        self.pressed = False
        self.x = []
        self.y = []
        self.line, = self.ax2.plot(self.x,self.y,linewidth=20,color='black')
        #self.clearButton = ButtonClickProcessor(self.ax1,self.ax2,self.ax3)
        self.connect()

    def clear(self,event):
        self.ax2.clear()
        self.ax3.clear()
        
    def connect(self):
        self.cidpress = self.fig.canvas.mpl_connect('button_press_event',self.onPress)
        self.cidrelease = self.fig.canvas.mpl_connect('button_release_event',self.onRelease)
        self.cidmove = self.fig.canvas.mpl_connect('motion_notify_event',self.onMove)

    def onPress(self, event):
        self.pressed = True
        self.ax3.clear()
        self.ax3.set_xticks([])
        self.ax3.set_yticks([])
        
    def onRelease(self, event):
        self.pressed = False
        self.fig.savefig('number.png',cmap='gray')
        self.x = []
        self.y = []
        self.ax2.set_axis_off()
        extent = self.ax2.get_window_extent().transformed(self.fig.dpi_scale_trans.inverted())
        self.fig.savefig('number.png', bbox_inches=extent)
        self.ax2.set_axis_on()
        self.ax3.clear()
        self.ax3.set_xticks([])
        self.ax3.set_yticks([])
        self.plotPrediction()

    def onMove(self, event):
        if self.pressed is True:
            self.x.append(event.xdata)
            self.y.append(event.ydata)
            self.line.set_data(self.x,self.y)
            self.line.figure.canvas.draw()

    def disconnect(self):
        self.fig.canvas.mpl_disconnect(self.cidpress)
        self.fig.canvas.mpl_disconnect(self.cidrelease)
        self.fig.canvas.mpl_disconnect(self.cidmove)

    def render(self):
        self.fig.canvas.draw()
        plt.show()

    def plotPrediction(self):
        drawn_image, predicted_num = predictNumber()
        self.ax3.text(0.5,0.5,str(predicted_num[0]), size=100, va="center", ha="center")
        self.ax3.set_xticks([])
        self.ax3.set_yticks([])
        self.render()
        
################################################################################

def main():
    digitRecognizer = DigitRecognizer()
    digitRecognizer.render()

################################################################################

main()
