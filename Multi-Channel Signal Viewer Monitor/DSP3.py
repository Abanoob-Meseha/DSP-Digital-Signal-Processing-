# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'DSP3.ui'
#
# Created by: PyQt5 UI code generator 5.15.4
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.

import sys
import random
import matplotlib
import pandas as pd
from PyQt5 import QtCore, QtWidgets
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from PIL import Image
import numpy as np
from PyQt5 import QtCore, QtGui, QtWidgets
from pyqtgraph import PlotWidget


class MplCanvas(FigureCanvas):

    """Ultimately, this is a QWidget (as well as a FigureCanvasAgg, etc.)."""

    def __init__(self, parent=None, width=5, height=5, dpi=100):

        fig = Figure(figsize=(width, height), dpi=dpi)
        self.figure = plt.Figure()
        super().__init__(fig)
        self.axes = fig.add_subplot(111)
        FigureCanvas.__init__(self, fig)
        self.setParent(parent)

        FigureCanvas.setSizePolicy(self,QtWidgets.QSizePolicy.Expanding,QtWidgets.QSizePolicy.Expanding)
        FigureCanvas.updateGeometry(self)
        


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(1425, 917)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.PLAY = QtWidgets.QPushButton(self.centralwidget)
        self.PLAY.setGeometry(QtCore.QRect(50, 670, 101, 51))
        font = QtGui.QFont()
        font.setPointSize(9)
        font.setBold(True)
        font.setWeight(75)
        self.PLAY.setFont(font)
        self.PLAY.setObjectName("PLAY")
        self.PAUSE = QtWidgets.QPushButton(self.centralwidget)
        self.PAUSE.setGeometry(QtCore.QRect(50, 740, 101, 51))
        font = QtGui.QFont()
        font.setPointSize(9)
        font.setBold(True)
        font.setWeight(75)
        self.PAUSE.setFont(font)
        self.PAUSE.setObjectName("PAUSE")
        self.ZOOMIN = QtWidgets.QPushButton(self.centralwidget)
        self.ZOOMIN.setGeometry(QtCore.QRect(190, 690, 101, 41))
        self.ZOOMIN.setObjectName("ZOOMIN")
        self.ZOOMOUT = QtWidgets.QPushButton(self.centralwidget)
        self.ZOOMOUT.setGeometry(QtCore.QRect(190, 740, 101, 41))
        self.ZOOMOUT.setObjectName("ZOOMOUT")
        self.SPECTRO_PLAY = QtWidgets.QPushButton(self.centralwidget)
        self.SPECTRO_PLAY.setGeometry(QtCore.QRect(1010, 729, 131, 61))
        font = QtGui.QFont()
        font.setPointSize(9)
        font.setBold(True)
        font.setWeight(75)
        self.SPECTRO_PLAY.setFont(font)
        self.SPECTRO_PLAY.setObjectName("SPECTRO_PLAY")
        self.EXPORT = QtWidgets.QPushButton(self.centralwidget)
        self.EXPORT.setGeometry(QtCore.QRect(1230, 20, 131, 51))
        self.EXPORT.setObjectName("EXPORT")
        self.SPEED_UP = QtWidgets.QPushButton(self.centralwidget)
        self.SPEED_UP.setGeometry(QtCore.QRect(320, 690, 101, 41))
        self.SPEED_UP.setObjectName("SPEED_UP")
        self.SPEED_DOWN = QtWidgets.QPushButton(self.centralwidget)
        self.SPEED_DOWN.setGeometry(QtCore.QRect(320, 740, 101, 41))
        self.SPEED_DOWN.setObjectName("SPEED_DOWN")
        self.HIDE = QtWidgets.QPushButton(self.centralwidget)
        self.HIDE.setGeometry(QtCore.QRect(450, 740, 81, 41))
        self.HIDE.setObjectName("HIDE")
        self.SHOW = QtWidgets.QPushButton(self.centralwidget)
        self.SHOW.setGeometry(QtCore.QRect(450, 690, 81, 41))
        self.SHOW.setObjectName("SHOW")
        self.SIGNAL_LABEL = QtWidgets.QLineEdit(self.centralwidget)
        self.SIGNAL_LABEL.setGeometry(QtCore.QRect(230, 70, 241, 31))
        self.SIGNAL_LABEL.setObjectName("SIGNAL_LABEL")
        self.label = QtWidgets.QLabel(self.centralwidget)
        self.label.setGeometry(QtCore.QRect(20, 70, 191, 31))
        font = QtGui.QFont()
        font.setPointSize(9)
        font.setBold(True)
        font.setWeight(75)
        self.label.setFont(font)
        self.label.setObjectName("label")
        self.ADD = QtWidgets.QPushButton(self.centralwidget)
        self.ADD.setGeometry(QtCore.QRect(500, 70, 81, 31))
        self.ADD.setObjectName("ADD")
        self.label_2 = QtWidgets.QLabel(self.centralwidget)
        self.label_2.setGeometry(QtCore.QRect(360, 130, 131, 41))
        font = QtGui.QFont()
        font.setPointSize(18)
        font.setBold(True)
        font.setUnderline(True)
        font.setWeight(75)
        self.label_2.setFont(font)
        self.label_2.setObjectName("label_2")
        self.label_3 = QtWidgets.QLabel(self.centralwidget)
        self.label_3.setGeometry(QtCore.QRect(1070, 120, 221, 41))
        font = QtGui.QFont()
        font.setPointSize(18)
        font.setBold(True)
        font.setUnderline(True)
        font.setWeight(75)
        self.label_3.setFont(font)
        self.label_3.setObjectName("label_3")
        self.SIGNALS = QtWidgets.QComboBox(self.centralwidget)
        self.SIGNALS.setGeometry(QtCore.QRect(20, 19, 141, 41))
        self.SIGNALS.setObjectName("SIGNALS")
        self.SIGNALS.addItem("")
        self.SIGNALS.addItem("")
        self.SIGNALS.addItem("")
        self.OPEN = QtWidgets.QPushButton(self.centralwidget)
        self.OPEN.setGeometry(QtCore.QRect(170, 20, 91, 41))
        self.OPEN.setObjectName("OPEN")
        self.SPECTROGRAMS = QtWidgets.QComboBox(self.centralwidget)
        self.SPECTROGRAMS.setGeometry(QtCore.QRect(980, 669, 191, 41))
        self.SPECTROGRAMS.setObjectName("SPECTROGRAMS")
        self.SPECTROGRAMS.addItem("")
        self.SPECTROGRAMS.addItem("")
        self.SPECTROGRAMS.addItem("")
        self.verticalSlider_MIN = QtWidgets.QSlider(self.centralwidget)
        self.verticalSlider_MIN.setGeometry(QtCore.QRect(1180, 659, 22, 160))
        self.verticalSlider_MIN.setOrientation(QtCore.Qt.Vertical)
        self.verticalSlider_MIN.setObjectName("verticalSlider_MIN")
        self.verticalSlider_MIN.setMinimum(-130)
        self.verticalSlider_MIN.setMaximum(-50)
        self.verticalSlider_MIN.setSingleStep(10)
        self.verticalSlider_MIN.setValue(-130)
        self.verticalSlider_MIN.setTickPosition(QtWidgets.QSlider.TicksLeft)
        self.verticalSlider_MIN.setTickInterval(10)
        self.verticalSlider_MAX = QtWidgets.QSlider(self.centralwidget)
        self.verticalSlider_MAX.setGeometry(QtCore.QRect(1220, 659, 22, 160))
        self.verticalSlider_MAX.setOrientation(QtCore.Qt.Vertical)
        self.verticalSlider_MAX.setObjectName("verticalSlider_MAX")
        self.verticalSlider_MAX.setMinimum(-50)
        self.verticalSlider_MAX.setMaximum(30)
        self.verticalSlider_MAX.setSingleStep(10)
        self.verticalSlider_MAX.setValue(30)
        self.verticalSlider_MAX.setTickPosition(QtWidgets.QSlider.TicksLeft)
        self.verticalSlider_MAX.setTickInterval(10)
        self.label_4 = QtWidgets.QLabel(self.centralwidget)
        self.label_4.setGeometry(QtCore.QRect(1180, 830, 31, 16))
        self.label_4.setObjectName("label_4")
        self.label_5 = QtWidgets.QLabel(self.centralwidget)
        self.label_5.setGeometry(QtCore.QRect(1220, 830, 31, 16))
        self.label_5.setObjectName("label_5")
        self.horizontalScrollBar = QtWidgets.QScrollBar(self.centralwidget)
        self.horizontalScrollBar.setGeometry(QtCore.QRect(30, 640, 881, 20))
        self.horizontalScrollBar.setOrientation(QtCore.Qt.Horizontal)
        self.horizontalScrollBar.setObjectName("horizontalScrollBar")
        self.horizontalB_bar_limit = 1000  
        self.horizontalScrollBar.setRange(0, self.horizontalB_bar_limit)


        self.COLOR_PALETTE = QtWidgets.QComboBox(self.centralwidget)
        self.COLOR_PALETTE.setGeometry(QtCore.QRect(1260, 670, 91, 41))
        self.COLOR_PALETTE.setObjectName("COLOR_PALETTE")
        self.COLOR_PALETTE.addItem("")
        self.COLOR_PALETTE.addItem("")
        self.COLOR_PALETTE.addItem("")
        self.COLOR_PALETTE.addItem("")
        self.COLOR_PALETTE.addItem("")
        self.CLEAR = QtWidgets.QPushButton(self.centralwidget)
        self.CLEAR.setGeometry(QtCore.QRect(50, 810, 101, 51))
        font = QtGui.QFont()
        font.setPointSize(9)
        font.setBold(True)
        font.setWeight(75)
        self.CLEAR.setFont(font)
        self.CLEAR.setObjectName("CLEAR")
        self.verticalScrollBar = QtWidgets.QScrollBar(self.centralwidget)
        self.verticalScrollBar.setGeometry(QtCore.QRect(10, 180, 20, 461))
        self.verticalScrollBar.setOrientation(QtCore.Qt.Vertical)
        self.verticalScrollBar.setObjectName("verticalScrollBar")
        self.Vertical_bar_limit = 1000 
        self.verticalScrollBar.setRange(0, self.Vertical_bar_limit) 


        self.COLORS = QtWidgets.QComboBox(self.centralwidget)
        self.COLORS.setGeometry(QtCore.QRect(570, 710, 131, 41))
        self.COLORS.setObjectName("COLORS")
        self.COLORS.addItem("")
        self.COLORS.addItem("")
        self.COLORS.addItem("")
        self.SETCOLOR = QtWidgets.QPushButton(self.centralwidget)
        self.SETCOLOR.setGeometry(QtCore.QRect(720, 710, 91, 41))
        self.SETCOLOR.setObjectName("SETCOLOR")
        self.line = QtWidgets.QFrame(self.centralwidget)
        self.line.setGeometry(QtCore.QRect(910, 660, 20, 201))
        self.line.setFrameShape(QtWidgets.QFrame.VLine)
        self.line.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line.setObjectName("line")
        self.splitter = QtWidgets.QSplitter(self.centralwidget)
        self.splitter.setGeometry(QtCore.QRect(30, 180, 1381, 461))
        self.splitter.setOrientation(QtCore.Qt.Horizontal)
        self.splitter.setObjectName("splitter")
        self.verticalLayoutWidget = QtWidgets.QWidget(self.splitter)
        self.verticalLayoutWidget.setObjectName("verticalLayoutWidget")
        self.verticalLayout = QtWidgets.QVBoxLayout(self.verticalLayoutWidget)
        self.verticalLayout.setContentsMargins(0, 0, 0, 0)
        self.verticalLayout.setObjectName("verticalLayout")
        self.graphicsView = PlotWidget(self.verticalLayoutWidget)
        self.graphicsView.setObjectName("graphicsView")
        self.verticalLayout.addWidget(self.graphicsView)
        self.verticalLayoutWidget_2 = QtWidgets.QWidget(self.splitter)
        self.verticalLayoutWidget_2.setObjectName("verticalLayoutWidget_2")
        self.verticalLayout_3 = QtWidgets.QVBoxLayout(self.verticalLayoutWidget_2)
        self.verticalLayout_3.setContentsMargins(10, 0, 0, 0)
        self.verticalLayout_3.setObjectName("verticalLayout_3")

        self.sc_1 = MplCanvas(self.verticalLayoutWidget_2, width=5, height=5, dpi=100)
        self.verticalLayout_3.addWidget(self.sc_1)
        
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 1425, 26))
        self.menubar.setObjectName("menubar")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.PLAY.setText(_translate("MainWindow", "PLAY"))
        self.PAUSE.setText(_translate("MainWindow", "PAUSE"))
        self.ZOOMIN.setText(_translate("MainWindow", "ZOOM IN"))
        self.ZOOMOUT.setText(_translate("MainWindow", "ZOOM OUT"))
        self.SPECTRO_PLAY.setText(_translate("MainWindow", "SPECTROGRAM"))
        self.EXPORT.setText(_translate("MainWindow", "EXPORT TO PDF"))
        self.SPEED_UP.setText(_translate("MainWindow", "SPEED UP"))
        self.SPEED_DOWN.setText(_translate("MainWindow", "SPEED DOWN"))
        self.HIDE.setText(_translate("MainWindow", "HIDE"))
        self.SHOW.setText(_translate("MainWindow", "SHOW"))
        self.label.setText(_translate("MainWindow", "ADD LABEL FOR SIGNAL :"))
        self.ADD.setText(_translate("MainWindow", "ADD"))
        self.label_2.setText(_translate("MainWindow", "SIGNAL"))
        self.label_3.setText(_translate("MainWindow", "SPECTOGRAM"))
        self.SIGNALS.setItemText(0, _translate("MainWindow", "SIGNAL 1"))
        self.SIGNALS.setItemText(1, _translate("MainWindow", "SIGNAL 2"))
        self.SIGNALS.setItemText(2, _translate("MainWindow", "SIGNAL 3"))
        self.OPEN.setText(_translate("MainWindow", "OPEN FILE"))
        self.SPECTROGRAMS.setItemText(0, _translate("MainWindow", "SIGNAL 1"))
        self.SPECTROGRAMS.setItemText(1, _translate("MainWindow", "SIGNAL 2"))
        self.SPECTROGRAMS.setItemText(2, _translate("MainWindow", "SIGNAL 3"))
        self.label_4.setText(_translate("MainWindow", "MIN"))
        self.label_5.setText(_translate("MainWindow", "MAX"))
        self.COLOR_PALETTE.setItemText(0, _translate("MainWindow", "viridis"))
        self.COLOR_PALETTE.setItemText(1, _translate("MainWindow", "plasma"))
        self.COLOR_PALETTE.setItemText(2, _translate("MainWindow", "inferno"))
        self.COLOR_PALETTE.setItemText(3, _translate("MainWindow", "magma"))
        self.COLOR_PALETTE.setItemText(4, _translate("MainWindow", "cividis"))
        self.CLEAR.setText(_translate("MainWindow", "CLEAR"))
        self.COLORS.setItemText(0, _translate("MainWindow", "BLUE"))
        self.COLORS.setItemText(1, _translate("MainWindow", "GREEN"))
        self.COLORS.setItemText(2, _translate("MainWindow", "RED"))
        self.SETCOLOR.setText(_translate("MainWindow", "SET COLOR"))


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())