#!/usr/bin/python3
# -*- coding: utf-8 -*-
##
import sys
import numpy as np
from PyQt5.QtWidgets import QWidget, QDesktopWidget, QTextEdit, QApplication, QDoubleSpinBox, QSpinBox, QGridLayout, QHBoxLayout, QVBoxLayout, QLabel, QSpacerItem, QSizePolicy, QPushButton
from PyQt5.QtCore import Qt, QObject, QThread, pyqtSignal, pyqtSlot
from PyQt5.QtGui import QCloseEvent, QPixmap, QImage
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import bezier
import time

current_milli_time = lambda: int(round(time.time() * 1000))

class Noise3D(QObject):
  
    def __init__(self, noise_depth=192, noise_height=192, noise_width=192):
        super().__init__()
        self.depth = noise_depth
        self.height = noise_height
        self.width = noise_width          
        self.noise = np.abs(np.random.randn(self.depth,self.height,self.width))
        
    def smooth(self, x, y, z):
        # get fractional part
        fractX = x - int(x)
        fractY = y - int(y)
        fractZ = z - int(z)
        # wrap around
        x1 = (int(x) + self.width) % self.width
        y1 = (int(y) + self.height) % self.height
        z1 = (int(z) + self.depth) % self.depth
        # neighbor values
        x2 = (x1 + self.width - 1) % self.width
        y2 = (y1 + self.height - 1) % self.height
        z2 = (z1 + self.depth - 1) % self.depth        
        # smooth the noise with bilinear interpolation
        value = 0.0
        value += fractX * fractY * fractZ * self.noise[z1,y1,x1]
        value += fractX * (1 - fractY) * fractZ * self.noise[z1,y2,x1]
        value += (1 - fractX) * fractY * fractZ * self.noise[z1,y1,x2]
        value += (1 - fractX) * (1 - fractY) * fractZ * self.noise[z1,y2,x2]

        value += fractX * fractY * (1 - fractZ) * self.noise[z2,y1,x1]
        value += fractX * (1 - fractY) * (1 - fractZ) * self.noise[z2,y2,x1]
        value += (1 - fractX) * fractY * (1 - fractZ) * self.noise[z2,y1,x2]
        value += (1 - fractX) * (1 - fractY) * (1 - fractZ) * self.noise[z2,y2,x2]
        return value

    def turbulence(self, x, y, z, size):
        value = 0.0
        initialSize = size
        while size >= 1.0:
            value += self.smooth(x / size, y / size, z / size) * size
            size /= 2.0

        return(128.0 * value / initialSize);      

class Tree:
  
    def __init__(self, data=np.array([0,0]), nrOfBranches=2):
        self.data = data
        self.brancheCounter = 0
        self.nrOfBranches = nrOfBranches
        self.branches = []
        
    def add(self, data=[0,0], nrOfBranches=2):
        if (self.brancheCounter<self.nrOfBranches):  # there is room
            self.branches.append(Tree(data, nrOfBranches))
            self.brancheCounter += 1
        else:
            print("Error: no more branches left")

    def getCurrBranche(self):
        if (self.brancheCounter>0):
            return self.branches[self.brancheCounter-1]
        else:
            print("Error: no branches")

    # def printPath(self, path): # this doesn't work. path keeps appending, although in different recursion level...
    #     path.append(self.data)
    #     print(self.data)
    #     if not self.branches:  # this is a leaf            
    #         print(np.asarray(path))
    #     else:
    #         for i in range(self.brancheCounter):
    #             self.branches[i].plotPath(path)

    def printPaths(self, root):
        if not root.branches:  # this is a leaf            
            return [str(root.data)] # [root.data] # 
        else:
            full_subtree = []
            for i in range(root.brancheCounter):
                full_subtree += self.printPaths(root.branches[i])
            list1 = []
            for leaf in full_subtree:  # middle part of the comprehension
##                list1.append([root.data, leaf])
                list1.append(str(root.data) + ',' + leaf)  # the left part
            return list1

    def BezierPlot(self, axis):
        brancheStr = self.printPaths(self)
        for i in range(len(brancheStr)):  # parse the list of tree path strings
            nodeStr = brancheStr[i].split(",")
            brancheLength = len(nodeStr)
            points = np.empty((brancheLength,2))
            for j in range(brancheLength):
                points[j,] = np.fromstring(nodeStr[j].translate({ord(ch): None for ch in '[]'}), dtype=float, sep=' ')
            nodes = np.asfortranarray(points.transpose())
            degree = nodes.shape[-1] - 1
            curve = bezier.Curve(nodes, degree=degree)
            greyVal = np.random.rand()
            curve.plot(num_pts=25, ax=axis, color= (greyVal,greyVal,greyVal))

    def traversePlot(self, axis):
        for i in range(self.brancheCounter):
            axis.plot((self.data[0],self.branches[i].data[0]), (self.data[1],self.branches[i].data[1]), color=(0,0,0), linewidth=1)
            self.branches[i].traversePlot(axis) 

    def getDepth(self):
        depth=0
        node=self
        while node.brancheCounter>0:
            depth += 1
            node = node.branches[0]
        return depth


class WorkerThread(QThread):    
    sigMsg = pyqtSignal(str)  # message to be shown to user
    treeReady = pyqtSignal(Tree)
    thread_name = "worker"
    
    def __init__(self, maxRecursions=1, angle=.25, shrink=1.3, nrOfBranches=2, noiseToroidRadius=20, noiseToroidPhaseInc=0.05):
        super().__init__()
        self.maxRecursions = maxRecursions
        self.angle = angle
        self.shrink = shrink
        self.stemLength = 2
        self.nrOfBranches = nrOfBranches
        self.noiseToroidRadius = noiseToroidRadius
        self.noiseToroidPhase = 0
        self.noiseToroidPhaseInc = noiseToroidPhaseInc
        self.noise = Noise3D(2*self.noiseToroidRadius,2*self.noiseToroidRadius,2*self.noiseToroidRadius)
        
    def run(self):
        try:
            # toroid through noise space
            self.x = 0.25*self.noiseToroidRadius*(1+np.cos(2*np.pi*self.noiseToroidPhase/self.noiseToroidRadius))*(1+np.cos(2*np.pi*self.noiseToroidPhase/self.noiseToroidRadius))
            self.y = 0.25*self.noiseToroidRadius*(1+np.sin(2*np.pi*self.noiseToroidPhase/self.noiseToroidRadius))*(1+np.sin(2*np.pi*self.noiseToroidPhase/self.noiseToroidRadius))
            self.z = 0.5*self.noiseToroidRadius*(1+np.sin(2*np.pi*self.noiseToroidPhase/self.noiseToroidRadius))

            tree = Tree(nrOfBranches=self.nrOfBranches)
            self.recursion(tree,[0,1],self.stemLength,0)
            self.treeReady.emit(tree)
####            self.sigMsg.emit("Image generated by " + self.thread_name)
            self.noiseToroidPhase += self.noiseToroidPhaseInc
        except Exception as err:
            self.sigMsg.emit(self.thread_name + ": Error, " + str(err))

    def recursion(self, tree, currGrad, size, n=0):
##        noiseValue = self.noise.turbulence(self.x+n, self.y+n, self.z, 2)
##        size *= (noiseValue/200)
##        newNode = np.add(tree.data, np.multiply(-size,currGrad))
##        tree.add(newNode, tree.nrOfBranches)
        newNode = np.add(tree.data, np.multiply(size,currGrad))
        tree.add(newNode, tree.nrOfBranches)
        if (n < self.maxRecursions): 
            for i in range(tree.nrOfBranches):
                noiseValue = self.noise.turbulence(self.x+n+i, self.y+n+i, self.z, 2)
                angle = self.angle + noiseValue/1000
                angle *= (i+1)*(-1)**i
                c, s = np.cos(angle), np.sin(angle)
                R = np.array(((c,-s), (s, c)))  # rotation matrix
                newGrad = np.dot(R, currGrad)
                self.recursion(tree.getCurrBranche(), newGrad, size/self.shrink, n+1)

    @pyqtSlot(float)
    def setAngle(self, n):
        self.angle = n
        self.start()

    @pyqtSlot(float)
    def setShrink(self, n):
        self.shrink = n
        self.start()

    @pyqtSlot(int)
    def setNrOfBranches(self, n):
        self.nrOfBranches = n
        self.start()

    @pyqtSlot(int)
    def setMaxRecursions(self, n):
        self.maxRecursions = n
        self.start()

    @pyqtSlot(float)
    def setNoiseToroidPhaseInc(self, n):
        self.noiseToroidPhaseInc = n
        self.start()
        
    def getAngle(self):
        return self.angle

    def getShrink(self):
        return self.shrink

    def getMaxRecursions(self):
        return self.maxRecursions    

    def getNoiseToroidPhaseInc(self):
        return self.noiseToroidPhaseInc
    

class LogWindow(QWidget):

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Log")
        self.move(100,100)
        layout = QVBoxLayout()
        self.setLayout(layout)
        self.resize(400, 800)
        self.log = QTextEdit()
        layout.addWidget(self.log)

    @pyqtSlot(str)
    def append(self, s):
        self.log.append(s)

        
class MainWindow(QWidget):
    closing = pyqtSignal()

    def __init__(self, maxRecursions=1, nrOfBranches=2, angle=.5, shrink=1.2, noiseToroidPhaseInc=0.1):
       super().__init__()
       self.image = None
       self.beta = 0.0
       self.maxRecursions = maxRecursions
       self.nrOfBranches = nrOfBranches
       self.angle = angle
       self.shrink = shrink
       self.noiseToroidPhaseInc = noiseToroidPhaseInc
       self.initUI()

    def initUI(self):
        self.setWindowTitle('Recursion tree')
        self.move(400,100)
        
        # Figure
        self.figure = plt.figure(figsize=(20,20))  # a figure instance to plot on
        self.figure.patch.set_alpha(0.5)  # transparent background
        # this is the Canvas Widget that displays the `figure`
        # it takes the `figure` instance as a parameter to __init__
        self.canvas = FigureCanvas(self.figure)

        # Button
        self.saveButton = QPushButton("Save")
        self.saveButton.clicked.connect(self.save)        
        
        # Spinboxes        
        self.maxRecursionsSpinBox = QSpinBox(self)
        self.maxRecursionsSpinBoxTitle = QLabel("Max recursions")
        self.maxRecursionsSpinBox.setMinimum(1)
        self.maxRecursionsSpinBox.setMaximum(999)
        self.maxRecursionsSpinBox.setValue(self.maxRecursions)        

        self.nrOfBranchesSpinBox = QSpinBox(self)
        self.nrOfBranchesSpinBoxTitle = QLabel("Nr of branches")
        self.nrOfBranchesSpinBox.setMinimum(1)
        self.nrOfBranchesSpinBox.setMaximum(9)
        self.nrOfBranchesSpinBox.setValue(self.nrOfBranches)

        self.shrinkSpinBox = QDoubleSpinBox(self)
        self.shrinkSpinBoxTitle = QLabel("Shrink")
        self.shrinkSpinBox.setSingleStep(0.01)
        self.shrinkSpinBox.setMinimum(0)
        self.shrinkSpinBox.setMaximum(5)
        self.shrinkSpinBox.setValue(self.shrink)

        self.angleSpinBox = QDoubleSpinBox(self)
        self.angleSpinBoxTitle = QLabel("Angle")
        self.angleSpinBox.setSingleStep(0.01)
        self.angleSpinBox.setMinimum(0)
        self.angleSpinBox.setMaximum(5)
        self.angleSpinBox.setValue(self.angle)

        self.betaSpinBox = QDoubleSpinBox(self)
        self.betaSpinBoxTitle = QLabel("Beta")
        self.betaSpinBox.setSingleStep(0.01)
        self.betaSpinBox.setMinimum(0.0)
        self.betaSpinBox.setMaximum(1.0)
        self.betaSpinBox.setValue(self.beta)
        self.betaSpinBox.valueChanged.connect(self.setBeta)        
        
        self.noiseToroidPhaseIncSpinBox = QDoubleSpinBox(self)
        self.noiseToroidPhaseIncSpinBoxTitle = QLabel("noiseToroidPhaseInc")
        self.noiseToroidPhaseIncSpinBox.setSingleStep(0.01)
        self.noiseToroidPhaseIncSpinBox.setMinimum(0.0)
        self.noiseToroidPhaseIncSpinBox.setMaximum(1.0)
        self.noiseToroidPhaseIncSpinBox.setValue(self.noiseToroidPhaseInc)

        # Compose layout grid
        keyWidgets = [self.nrOfBranchesSpinBoxTitle, self.maxRecursionsSpinBoxTitle, self.angleSpinBoxTitle, self.shrinkSpinBoxTitle,
                      self.betaSpinBoxTitle, self.noiseToroidPhaseIncSpinBoxTitle, self.saveButton]
        valueWidgets = [self.nrOfBranchesSpinBox, self.maxRecursionsSpinBox, self.angleSpinBox, self.shrinkSpinBox,
                        self.betaSpinBox, self.noiseToroidPhaseIncSpinBox, None]
        widgetLayout = QGridLayout()
        for index, widget in enumerate(keyWidgets):
            if widget is not None:
                widgetLayout.addWidget(widget, index, 0, Qt.AlignLeft)
        for index, widget in enumerate(valueWidgets):
            if widget is not None:
                widgetLayout.addWidget(widget, index, 1, Qt.AlignLeft)
        widgetLayout.setSpacing(10)
        widgetLayout.addItem(QSpacerItem(0, 0, QSizePolicy.Minimum,QSizePolicy.Expanding))  # variable space
        layout = QHBoxLayout()
        layout.addLayout(widgetLayout, Qt.AlignTop|Qt.AlignCenter)
        layout.addWidget(self.canvas, Qt.AlignTop|Qt.AlignCenter)
        layout.setSpacing(10)
        self.setLayout(layout)

    def save(self):
        plt.savefig('RecursionTreeBezier_'+ str(current_milli_time()) + '.svg')
        
    @pyqtSlot(Tree)
    def onTreeReady(self, tree=None):
        self.figure.clear()
        ax = self.figure.add_subplot(111)
        # tree.traversePlot(ax)
        tree.BezierPlot(ax)
        plt.axis('off')
##        plt.xlim(-6, 6)
##        plt.ylim(-2, 8)
        ax.grid(False)
        self.canvas.draw()  # refresh canvas

    @pyqtSlot(float)
    def setBeta(self, n):
        self.beta = n
        
    def closeEvent(self, event: QCloseEvent):
        self.closing.emit()
        event.accept()


if __name__ == '__main__':
    app = QApplication(sys.argv)
    
    # Instantiate objects
    worker = WorkerThread(noiseToroidRadius=10)
    mainWindow = MainWindow(maxRecursions=worker.getMaxRecursions(),
                            angle=worker.getAngle(),
                            shrink=worker.getShrink())
    logWindow = LogWindow()
    
    # Connect signals and slots    
    worker.treeReady.connect(mainWindow.onTreeReady)  # Post worker's result to main Window
    worker.sigMsg.connect(logWindow.append)  # Log worker messages
    mainWindow.closing.connect(worker.quit)  # Quit worker thread
    mainWindow.closing.connect(logWindow.close)  # Close log window
##    mainWindow.closing.connect(app.exit)  # Close app
    mainWindow.nrOfBranchesSpinBox.valueChanged.connect(worker.setNrOfBranches)
    mainWindow.maxRecursionsSpinBox.valueChanged.connect(worker.setMaxRecursions)
    mainWindow.shrinkSpinBox.valueChanged.connect(worker.setShrink)
    mainWindow.angleSpinBox.valueChanged.connect(worker.setAngle)    
##    mainWindow.noiseToroidPhaseIncSpinBox.valueChanged.connect(worker.setNoiseToroidPhaseInc)    

    # Start the show
    worker.start()
    logWindow.show()
    mainWindow.show()
    # sys.exit(app.exec_())
    app.exec_()
    
