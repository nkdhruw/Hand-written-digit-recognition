import matplotlib.pyplot as plt
import numpy as np

########################################################################################

class Pencil:
    def __init__(self,fig,ax):
        self.fig = fig
        self.pressed = False
        self.x = []
        self.y = []
        self.line, = ax.plot(self.x,self.y,linewidth=20,color='black')

    def connect(self):
        self.cidpress = self.fig.canvas.mpl_connect('button_press_event',self.onPress)
        self.cidrelease = self.fig.canvas.mpl_connect('button_release_event',self.onRelease)
        self.cidmove = self.fig.canvas.mpl_connect('motion_notify_event',self.onMove)

    def onPress(self, event):
        self.pressed = True
        print('Mouse button pressed')

    def onRelease(self, event):
        self.pressed = False
        self.fig.savefig('number.png',cmap='gray')
        self.x = []
        self.y = []
        print('Mouse button released')

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

########################################################################################

def main():
    fig = plt.figure(frameon=False)
    fig.set_size_inches(3,3)
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)
    pencil = Pencil(fig, ax)
    pencil.connect()
    plt.show()

########################################################################################

if __name__ =="__main__":
    main()
