# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'DSP2FINAL.ui'
#
# Created by: PyQt5 UI code generator 5.15.4
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets
from pyqtgraph import PlotWidget


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(1093, 856)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.gridLayout_2 = QtWidgets.QGridLayout(self.centralwidget)
        self.gridLayout_2.setObjectName("gridLayout_2")
        self.tabWidget = QtWidgets.QTabWidget(self.centralwidget)
        self.tabWidget.setMovable(False)
        self.tabWidget.setObjectName("tabWidget")
        self.SAMPLING = QtWidgets.QWidget()
        self.SAMPLING.setObjectName("SAMPLING")
        self.gridLayout = QtWidgets.QGridLayout(self.SAMPLING)
        self.gridLayout.setObjectName("gridLayout")
        self.CONSTRUCT = QtWidgets.QPushButton(self.SAMPLING)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Maximum)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.CONSTRUCT.sizePolicy().hasHeightForWidth())
        self.CONSTRUCT.setSizePolicy(sizePolicy)
        self.CONSTRUCT.setObjectName("CONSTRUCT")
        self.gridLayout.addWidget(self.CONSTRUCT, 6, 3, 1, 2)
        self.MAIN_COLOR = QtWidgets.QComboBox(self.SAMPLING)
        self.MAIN_COLOR.setObjectName("MAIN_COLOR")
        self.MAIN_COLOR.addItem("")
        self.MAIN_COLOR.addItem("")
        self.MAIN_COLOR.addItem("")
        self.gridLayout.addWidget(self.MAIN_COLOR, 1, 2, 1, 1)
        self.verticalLayout = QtWidgets.QVBoxLayout()
        self.verticalLayout.setObjectName("verticalLayout")
        self.graphicsView = PlotWidget(self.SAMPLING)
        self.graphicsView.setObjectName("graphicsView")
        self.verticalLayout.addWidget(self.graphicsView)
        self.graphicsView_2 = PlotWidget(self.SAMPLING)
        self.graphicsView_2.setObjectName("graphicsView_2")
        self.verticalLayout.addWidget(self.graphicsView_2)
        self.gridLayout.addLayout(self.verticalLayout, 0, 0, 11, 1)
        self.label = QtWidgets.QLabel(self.SAMPLING)
        font = QtGui.QFont()
        font.setPointSize(9)
        font.setBold(True)
        font.setWeight(75)
        self.label.setFont(font)
        self.label.setObjectName("label")
        self.gridLayout.addWidget(self.label, 4, 1, 1, 1)
        self.MIGRATE = QtWidgets.QPushButton(self.SAMPLING)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Maximum)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.MIGRATE.sizePolicy().hasHeightForWidth())
        self.MIGRATE.setSizePolicy(sizePolicy)
        self.MIGRATE.setObjectName("MIGRATE")
        self.gridLayout.addWidget(self.MIGRATE, 9, 4, 1, 1)
        self.LOAD = QtWidgets.QPushButton(self.SAMPLING)
        self.LOAD.setObjectName("LOAD")
        self.gridLayout.addWidget(self.LOAD, 1, 3, 1, 2)
        self.SAMPLING_RATE = QtWidgets.QSpinBox(self.SAMPLING)
        self.SAMPLING_RATE.setObjectName("SAMPLING_RATE")
        self.gridLayout.addWidget(self.SAMPLING_RATE, 4, 2, 1, 2)
        self.SAMPLING_COLOR = QtWidgets.QComboBox(self.SAMPLING)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Maximum)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.SAMPLING_COLOR.sizePolicy().hasHeightForWidth())
        self.SAMPLING_COLOR.setSizePolicy(sizePolicy)
        self.SAMPLING_COLOR.setObjectName("SAMPLING_COLOR")
        self.SAMPLING_COLOR.addItem("")
        self.SAMPLING_COLOR.addItem("")
        self.SAMPLING_COLOR.addItem("")
        self.gridLayout.addWidget(self.SAMPLING_COLOR, 5, 2, 1, 1)
        self.SAMPLE = QtWidgets.QPushButton(self.SAMPLING)
        self.SAMPLE.setObjectName("SAMPLE")
        self.gridLayout.addWidget(self.SAMPLE, 5, 3, 1, 2)
        self.line_2 = QtWidgets.QFrame(self.SAMPLING)
        self.line_2.setFrameShape(QtWidgets.QFrame.HLine)
        self.line_2.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line_2.setObjectName("line_2")
        self.gridLayout.addWidget(self.line_2, 7, 1, 1, 4)
        self.line = QtWidgets.QFrame(self.SAMPLING)
        self.line.setFrameShape(QtWidgets.QFrame.HLine)
        self.line.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line.setObjectName("line")
        self.gridLayout.addWidget(self.line, 2, 1, 1, 4)
        self.label_3 = QtWidgets.QLabel(self.SAMPLING)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Maximum)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.label_3.sizePolicy().hasHeightForWidth())
        self.label_3.setSizePolicy(sizePolicy)
        font = QtGui.QFont()
        font.setPointSize(10)
        font.setBold(True)
        font.setUnderline(True)
        font.setWeight(75)
        self.label_3.setFont(font)
        self.label_3.setObjectName("label_3")
        self.gridLayout.addWidget(self.label_3, 3, 4, 1, 1)
        self.HIDE = QtWidgets.QPushButton(self.SAMPLING)
        self.HIDE.setObjectName("HIDE")
        self.gridLayout.addWidget(self.HIDE, 10, 4, 1, 1)
        self.label_4 = QtWidgets.QLabel(self.SAMPLING)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Maximum)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.label_4.sizePolicy().hasHeightForWidth())
        self.label_4.setSizePolicy(sizePolicy)
        font = QtGui.QFont()
        font.setPointSize(10)
        font.setBold(True)
        font.setUnderline(True)
        font.setWeight(75)
        self.label_4.setFont(font)
        self.label_4.setObjectName("label_4")
        self.gridLayout.addWidget(self.label_4, 8, 4, 1, 1)
        self.label_2 = QtWidgets.QLabel(self.SAMPLING)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Maximum)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.label_2.sizePolicy().hasHeightForWidth())
        self.label_2.setSizePolicy(sizePolicy)
        font = QtGui.QFont()
        font.setPointSize(10)
        font.setBold(True)
        font.setUnderline(True)
        font.setWeight(75)
        self.label_2.setFont(font)
        self.label_2.setObjectName("label_2")
        self.gridLayout.addWidget(self.label_2, 0, 4, 1, 1)
        self.RESET = QtWidgets.QPushButton(self.SAMPLING)
        self.RESET.setObjectName("RESET")
        self.gridLayout.addWidget(self.RESET, 4, 4, 1, 1)
        self.tabWidget.addTab(self.SAMPLING, "")
        self.COMPOSER = QtWidgets.QWidget()
        self.COMPOSER.setObjectName("COMPOSER")
        self.gridLayout_3 = QtWidgets.QGridLayout(self.COMPOSER)
        self.gridLayout_3.setObjectName("gridLayout_3")
        self.EXPORT = QtWidgets.QPushButton(self.COMPOSER)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Maximum)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.EXPORT.sizePolicy().hasHeightForWidth())
        self.EXPORT.setSizePolicy(sizePolicy)
        self.EXPORT.setObjectName("EXPORT")
        self.gridLayout_3.addWidget(self.EXPORT, 7, 1, 1, 2)
        self.PHASE = QtWidgets.QDoubleSpinBox(self.COMPOSER)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Maximum)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.PHASE.sizePolicy().hasHeightForWidth())
        self.PHASE.setSizePolicy(sizePolicy)
        self.PHASE.setObjectName("PHASE")
        self.gridLayout_3.addWidget(self.PHASE, 1, 2, 1, 1)
        self.MAGNITUDE = QtWidgets.QDoubleSpinBox(self.COMPOSER)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Maximum)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.MAGNITUDE.sizePolicy().hasHeightForWidth())
        self.MAGNITUDE.setSizePolicy(sizePolicy)
        self.MAGNITUDE.setMinimumSize(QtCore.QSize(7, 8))
        self.MAGNITUDE.setObjectName("MAGNITUDE")
        self.gridLayout_3.addWidget(self.MAGNITUDE, 0, 2, 1, 1)
        self.label_5 = QtWidgets.QLabel(self.COMPOSER)
        font = QtGui.QFont()
        font.setPointSize(9)
        font.setBold(True)
        font.setWeight(75)
        self.label_5.setFont(font)
        self.label_5.setObjectName("label_5")
        self.gridLayout_3.addWidget(self.label_5, 0, 1, 1, 1)
        self.label_6 = QtWidgets.QLabel(self.COMPOSER)
        font = QtGui.QFont()
        font.setPointSize(10)
        font.setBold(True)
        font.setWeight(75)
        self.label_6.setFont(font)
        self.label_6.setObjectName("label_6")
        self.gridLayout_3.addWidget(self.label_6, 1, 1, 1, 1)
        self.ADD = QtWidgets.QPushButton(self.COMPOSER)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Maximum)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.ADD.sizePolicy().hasHeightForWidth())
        self.ADD.setSizePolicy(sizePolicy)
        self.ADD.setObjectName("ADD")
        self.gridLayout_3.addWidget(self.ADD, 3, 2, 1, 1)
        self.FREQUENCY = QtWidgets.QDoubleSpinBox(self.COMPOSER)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Maximum)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.FREQUENCY.sizePolicy().hasHeightForWidth())
        self.FREQUENCY.setSizePolicy(sizePolicy)
        self.FREQUENCY.setObjectName("FREQUENCY")
        self.gridLayout_3.addWidget(self.FREQUENCY, 2, 2, 1, 1)
        self.CONFIRM = QtWidgets.QPushButton(self.COMPOSER)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Maximum)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.CONFIRM.sizePolicy().hasHeightForWidth())
        self.CONFIRM.setSizePolicy(sizePolicy)
        self.CONFIRM.setObjectName("CONFIRM")
        self.gridLayout_3.addWidget(self.CONFIRM, 5, 1, 1, 2)
        self.line_3 = QtWidgets.QFrame(self.COMPOSER)
        self.line_3.setFrameShape(QtWidgets.QFrame.HLine)
        self.line_3.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line_3.setObjectName("line_3")
        self.gridLayout_3.addWidget(self.line_3, 6, 1, 1, 2)
        self.label_7 = QtWidgets.QLabel(self.COMPOSER)
        font = QtGui.QFont()
        font.setPointSize(9)
        font.setBold(True)
        font.setWeight(75)
        self.label_7.setFont(font)
        self.label_7.setObjectName("label_7")
        self.gridLayout_3.addWidget(self.label_7, 2, 1, 1, 1)
        self.splitter_2 = QtWidgets.QSplitter(self.COMPOSER)
        self.splitter_2.setOrientation(QtCore.Qt.Vertical)
        self.splitter_2.setObjectName("splitter_2")
        self.verticalLayoutWidget_3 = QtWidgets.QWidget(self.splitter_2)
        self.verticalLayoutWidget_3.setObjectName("verticalLayoutWidget_3")
        self.verticalLayout_3 = QtWidgets.QVBoxLayout(self.verticalLayoutWidget_3)
        self.verticalLayout_3.setSizeConstraint(QtWidgets.QLayout.SetDefaultConstraint)
        self.verticalLayout_3.setContentsMargins(0, 0, 0, 0)
        self.verticalLayout_3.setObjectName("verticalLayout_3")
        self.graphicsView_3 = PlotWidget(self.verticalLayoutWidget_3)
        self.graphicsView_3.setObjectName("graphicsView_3")
        self.verticalLayout_3.addWidget(self.graphicsView_3)
        self.verticalLayoutWidget_4 = QtWidgets.QWidget(self.splitter_2)
        self.verticalLayoutWidget_4.setObjectName("verticalLayoutWidget_4")
        self.verticalLayout_4 = QtWidgets.QVBoxLayout(self.verticalLayoutWidget_4)
        self.verticalLayout_4.setContentsMargins(0, 0, 0, 0)
        self.verticalLayout_4.setObjectName("verticalLayout_4")
        self.graphicsView_4 = PlotWidget(self.verticalLayoutWidget_4)
        self.graphicsView_4.setObjectName("graphicsView_4")
        self.verticalLayout_4.addWidget(self.graphicsView_4)
        self.gridLayout_3.addWidget(self.splitter_2, 0, 0, 9, 1)
        self.MOVE_TO_MAIN = QtWidgets.QPushButton(self.COMPOSER)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Maximum)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.MOVE_TO_MAIN.sizePolicy().hasHeightForWidth())
        self.MOVE_TO_MAIN.setSizePolicy(sizePolicy)
        font = QtGui.QFont()
        font.setPointSize(10)
        font.setBold(True)
        font.setUnderline(True)
        font.setWeight(75)
        self.MOVE_TO_MAIN.setFont(font)
        self.MOVE_TO_MAIN.setObjectName("MOVE_TO_MAIN")
        self.gridLayout_3.addWidget(self.MOVE_TO_MAIN, 8, 1, 1, 2)
        self.DELETE = QtWidgets.QPushButton(self.COMPOSER)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Maximum)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.DELETE.sizePolicy().hasHeightForWidth())
        self.DELETE.setSizePolicy(sizePolicy)
        self.DELETE.setObjectName("DELETE")
        self.gridLayout_3.addWidget(self.DELETE, 4, 2, 1, 1)
        self.SIGNALS = QtWidgets.QComboBox(self.COMPOSER)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Maximum, QtWidgets.QSizePolicy.Maximum)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.SIGNALS.sizePolicy().hasHeightForWidth())
        self.SIGNALS.setSizePolicy(sizePolicy)
        self.SIGNALS.setObjectName("SIGNALS")
        self.SIGNALS.addItem("")
        self.SIGNALS.addItem("")
        self.SIGNALS.addItem("")
        self.SIGNALS.addItem("")
        self.gridLayout_3.addWidget(self.SIGNALS, 3, 1, 2, 1)
        self.tabWidget.addTab(self.COMPOSER, "")
        self.gridLayout_2.addWidget(self.tabWidget, 0, 1, 1, 1)
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 1093, 26))
        self.menubar.setObjectName("menubar")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        self.tabWidget.setCurrentIndex(0)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "SIGNAL PROCESSING (SAMPLING)"))
        self.CONSTRUCT.setText(_translate("MainWindow", "CONSTRUCT"))
        self.MAIN_COLOR.setItemText(0, _translate("MainWindow", "RED"))
        self.MAIN_COLOR.setItemText(1, _translate("MainWindow", "GREEN"))
        self.MAIN_COLOR.setItemText(2, _translate("MainWindow", "BLUE"))
        self.label.setText(_translate("MainWindow", "SAMPLING RATE:"))
        self.MIGRATE.setText(_translate("MainWindow", "MIGRATE HERE"))
        self.LOAD.setText(_translate("MainWindow", "LOAD"))
        self.SAMPLING_COLOR.setItemText(0, _translate("MainWindow", "RED"))
        self.SAMPLING_COLOR.setItemText(1, _translate("MainWindow", "GREEN"))
        self.SAMPLING_COLOR.setItemText(2, _translate("MainWindow", "BLUE"))
        self.SAMPLE.setText(_translate("MainWindow", "SAMPLE"))
        self.label_3.setText(_translate("MainWindow", "SAMPLING"))
        self.HIDE.setText(_translate("MainWindow", "HIDE"))
        self.label_4.setText(_translate("MainWindow", "GRAPH 2 SPECS"))
        self.label_2.setText(_translate("MainWindow", "SIGNAL READ"))
        self.RESET.setText(_translate("MainWindow", "RESET"))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.SAMPLING), _translate("MainWindow", "SAMPLING"))
        self.EXPORT.setText(_translate("MainWindow", "EXPORT"))
        self.label_5.setText(_translate("MainWindow", "MAGNITUDE :"))
        self.label_6.setText(_translate("MainWindow", "PHASE :"))
        self.ADD.setText(_translate("MainWindow", "ADD"))
        self.CONFIRM.setText(_translate("MainWindow", "CONFIRM"))
        self.label_7.setText(_translate("MainWindow", "FREQUENCY :"))
        self.MOVE_TO_MAIN.setText(_translate("MainWindow", "MOVE TO MAIN"))
        self.DELETE.setText(_translate("MainWindow", "DELETE"))
        self.SIGNALS.setItemText(0, _translate("MainWindow", "SIGNAL 1"))
        self.SIGNALS.setItemText(1, _translate("MainWindow", "SIGNAL 2"))
        self.SIGNALS.setItemText(2, _translate("MainWindow", "SIGNAL 3"))
        self.SIGNALS.setItemText(3, _translate("MainWindow", "SIGNAL 4"))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.COMPOSER), _translate("MainWindow", "COMPOSER"))
