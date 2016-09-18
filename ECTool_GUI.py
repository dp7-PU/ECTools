import numpy as np
from PyQt4 import QtGui, QtCore
import sys
import os
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt4agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt4agg import \
    NavigationToolbar2QT as NavigationToolbar
import ECTools
import pandas as pd
from scipy.interpolate import LinearNDInterpolator
import pickle as pl


class mplCanvas(QtGui.QWidget):
    def __init__(self, parent=None, nrow=1, ncol=1, width=5, height=4, dpi=80,
                 bgcolor='#ffffff'):
        super(mplCanvas, self).__init__(parent)

        # a figure instance to plot on
        self.fig, self.axes = plt.subplots(nrow, ncol, figsize=(width, height),
                                           dpi=dpi, facecolor=bgcolor, sharex=True)
        for ax in self.axes:
            ax.hold(False)
        # this is the Canvas Widget that displays the `figure`
        # it takes the `figure` instance as a parameter to __init__
        self.fig.tight_layout()
        self.fig.subplots_adjust(hspace=0.12)
        self.canvas = FigureCanvas(self.fig)
        FigureCanvas.setSizePolicy(self.canvas,
                                   QtGui.QSizePolicy.Expanding,
                                   QtGui.QSizePolicy.Expanding)
        FigureCanvas.updateGeometry(self)
        # this is the Navigation widget
        # it takes the Canvas widget and a parent
        self.toolbar = NavigationToolbar(self.canvas, self)

        # set the layout
        layout = QtGui.QVBoxLayout()
        layout.addWidget(self.toolbar)
        layout.addWidget(self.canvas)
        self.setLayout(layout)


class AppWindow(QtGui.QMainWindow):
    def __init__(self, parent=None):
        super(AppWindow, self).__init__(parent)
        self.setWindowTitle('ECTools')
        self.scrsz = QtGui.QDesktopWidget().availableGeometry().getRect()
        self.dpi = int(self.scrsz[2] / 25)
        self.started = False
        self.initUI()

    def initUI(self):
        self.mainWidget = QtGui.QWidget()
        self.setCentralWidget(self.mainWidget)
        self.resize(0.8 * self.scrsz[2], 0.8 * self.scrsz[3])
        self.config = ECTools.read_config(settingFile)
        self.showMaximized()
        self.setFrames()
        self.setLeftTopFrame()
        self.setLeftMid()
        self.setLeftBottom()
        self.setMid()
        self.setRight()
        self.createAdvancedSettingDialog()

    def getFileDirDialog(self):
        DirName = QtGui.QFileDialog.getExistingDirectory(self)
        return DirName

    def setFrames(self):
        self.leftTop = QtGui.QGroupBox(self.mainWidget)
        self.leftTop.setTitle('Load files and calculate fluxes')

        self.leftMid = QtGui.QGroupBox(self.mainWidget)
        self.leftMid.setTitle('Cross-covariance plot:')

        self.leftBottom = QtGui.QGroupBox(self.mainWidget)
        self.leftBottom.setTitle('Status log:')

        self.leftSplit = QtGui.QSplitter(QtCore.Qt.Vertical)
        self.leftSplit.addWidget(self.leftTop)
        self.leftSplit.addWidget(self.leftMid)
        self.leftSplit.addWidget(self.leftBottom)

        self.mid = QtGui.QGroupBox(self.mainWidget)
        self.mid.setTitle('Measurement time series')

        self.right = QtGui.QGroupBox(self.mainWidget)
        self.right.setTitle('30-min averaged results')

        self.split = QtGui.QSplitter(QtCore.Qt.Horizontal)
        self.split.addWidget(self.leftSplit)
        self.split.addWidget(self.mid)
        self.split.addWidget(self.right)
        self.split.setSizes(self.scrsz[2] * np.array([0.2, 0.4, 0.4]))

        hbox = QtGui.QHBoxLayout()
        hbox.addWidget(self.split)
        self.mainWidget.setLayout(hbox)

    def setLeftTopFrame(self):
        vbox = QtGui.QVBoxLayout()

        # QCL gas selection
        QCLGasLabel = QtGui.QLabel('QCL target gas:')
        self.QCLGasCombo = QtGui.QComboBox()
        self.QCLGasCombo.addItems(['N2O', 'NH3'])
        QCLGasHBox = QtGui.QHBoxLayout()
        QCLGasHBox.addWidget(QCLGasLabel)
        QCLGasHBox.addWidget(self.QCLGasCombo)

        # TOA5 Selection
        TOA5Label = QtGui.QLabel('Select TOA5 file:')
        TOA5Hbox = QtGui.QHBoxLayout()
        self.TOA5Str = QtGui.QLineEdit(self.config['TOA5_name'])
        TOA5Button = QtGui.QPushButton('Browse')
        TOA5Button.clicked.connect(self.getTOA5File)
        TOA5Hbox.addWidget(self.TOA5Str)
        TOA5Hbox.addWidget(TOA5Button)

        # Sensor raw file directory selection
        rawDirLabel = QtGui.QLabel('Select QCL raw data folder:')
        rawDirHbox = QtGui.QHBoxLayout()
        self.rawDirStr = QtGui.QLineEdit()
        self.rawDirStr.setText(self.config['dir'])
        self.rawDirStr.setMinimumWidth(250)
        rawDirButton = QtGui.QPushButton('Browse')
        rawDirButton.clicked.connect(self.getRawDir)
        rawDirHbox.addWidget(self.rawDirStr)
        rawDirHbox.addWidget(rawDirButton)

        # Set time span
        timeSpanLabel = QtGui.QLabel('Set start and end time:')
        startTimeHBox = QtGui.QHBoxLayout()
        endTimeHBox = QtGui.QHBoxLayout()
        self.startTimeStr = QtGui.QLineEdit(self.config['t_start'])
        startTimeButton = QtGui.QPushButton('Calendar')
        startTimeButton.clicked.connect(self.getStartTime)
        self.endTimeStr = QtGui.QLineEdit(self.config['t_end'])
        endTimeButton = QtGui.QPushButton('Calendar')
        endTimeButton.clicked.connect(self.getEndTime)
        startTimeHBox.addWidget(self.startTimeStr)
        startTimeHBox.addWidget(startTimeButton)
        endTimeHBox.addWidget(self.endTimeStr)
        endTimeHBox.addWidget(endTimeButton)

        # Show time-lag setting
        showLagLabel = QtGui.QLabel('Show time lag during calculation:')
        self.showLag = QtGui.QRadioButton()
        if self.config['show_lag'] == 'True':
            self.showLag.setChecked(True)
        else:
            self.showLag.setChecked(False)

        showLagHBox = QtGui.QHBoxLayout()
        showLagHBox.addWidget(showLagLabel)
        showLagHBox.addWidget(self.showLag)
        lagLimLabel = QtGui.QLabel('Lag time limit (sec):')
        self.lagLimStr = QtGui.QLineEdit()
        self.lagLimStr.setText(str(self.config['lag_lim']))
        showLagHBox.addWidget(lagLimLabel)
        showLagHBox.addWidget(self.lagLimStr)
        showLagText = QtGui.QLabel('*Check to see if QCL is'
                                   ' synchronized with anonemeter.')

        # Set save dir
        saveDirLabel = QtGui.QLabel('Save results to:')
        self.saveDir = QtGui.QLineEdit()
        saveButton = QtGui.QPushButton('Browse')
        saveButton.clicked.connect(self.getSaveDir)
        saveHBox = QtGui.QHBoxLayout()
        if 'save_dir' in self.config.keys():
            self.saveDir.setText(self.config['save_dir'])
        else:
            self.saveDir.setText(cwd)

        saveHBox.addWidget(self.saveDir)
        saveHBox.addWidget(saveButton)

        # Buttons
        ButtonsHBox = QtGui.QHBoxLayout()
        saveSetButton = QtGui.QPushButton('Save Settings')
        saveSetButton.clicked.connect(self.saveConfig)
        startButton = QtGui.QPushButton('Start')
        startButton.clicked.connect(self.calc_flux)
        stopButton = QtGui.QPushButton('Stop')
        stopButton.clicked.connect(self.stop_calc)
        advanceSetButton = QtGui.QPushButton('Advanced Settings')
        advanceSetButton.clicked.connect(self.showAdvancedSetDialog)
        ButtonsHBox.addWidget(advanceSetButton)
        ButtonsHBox.addWidget(saveSetButton)
        ButtonsHBox.addWidget(startButton)
        ButtonsHBox.addWidget(stopButton)

        vbox.addLayout(QCLGasHBox)
        vbox.addWidget(TOA5Label)
        vbox.addLayout(TOA5Hbox)
        vbox.addWidget(rawDirLabel)
        vbox.addLayout(rawDirHbox)
        vbox.addWidget(timeSpanLabel)
        vbox.addLayout(startTimeHBox)
        vbox.addLayout(endTimeHBox)
        vbox.addLayout(showLagHBox)
        vbox.addWidget(showLagText)
        # vbox.addLayout(lagLimHBox)
        vbox.addWidget(saveDirLabel)
        vbox.addLayout(saveHBox)
        vbox.addLayout(ButtonsHBox)

        vbox.setAlignment(QtCore.Qt.AlignTop)

        self.leftTop.setLayout(vbox)
        self.leftTop.setAlignment(QtCore.Qt.AlignTop)

    def setLeftMid(self):
        """
    
        Parameters
        ----------
    
        Returns
        -------
    
        """
        self.canvasXCov = mplCanvas(self, nrow=3, ncol=1)
        VBox = QtGui.QVBoxLayout()
        VBox.addWidget(self.canvasXCov)
        self.leftMid.setLayout(VBox)

    def setLeftBottom(self):
        self.statusBox = QtGui.QTextEdit(self)
        self.statusBox.setReadOnly(True)
        self.statusBox.setText('Waiting for inputs. ')
        VBox = QtGui.QVBoxLayout()
        VBox.addWidget(self.statusBox)
        self.leftBottom.setLayout(VBox)

    def setMid(self):
        self.canvasTS = mplCanvas(self, nrow=6, ncol=1)
        VBox = QtGui.QVBoxLayout()
        VBox.addWidget(self.canvasTS)
        self.mid.setLayout(VBox)

    def setRight(self):
        self.canvasEC = mplCanvas(self, nrow=6, ncol=1)
        VBox = QtGui.QVBoxLayout()
        VBox.addWidget(self.canvasEC)
        self.right.setLayout(VBox)

    def getTOA5File(self):
        TOA5File = self.getFileNameDiaglog()
        self.TOA5Str.setText(TOA5File)

    def getSpecCorrFile(self):
        specCorrFile = self.getFileNameDiaglog()
        self.config['spec_corr_file'] = specCorrFile
        self.specCorrStr.setText(specCorrFile)

    def getSetFile(self):
        setFile = self.getFileNameDiaglog()
        self.readSettingFromFile(setFile)

    def readSettingFromFile(self, setFile):
        self.config = ECTools.read_config(setFile)
        self.setFileStr.setText(setFile)
        self.colNameStr.setText(', '.join(self.config['col_names']))
        self.TOA5Str.setText(self.config['TOA5_name'])
        self.rawDirStr.setText(self.config['dir'])
        self.despikeLimStr.setText(', '.join(map(str, self.config['despike_lim'])))
        self.fileTypeStr.setText(self.config['file_type'])
        self.basetimeStr.setText(self.config['base_time'])
        self.prefixStr.setText(self.config['prefix'])
        self.specCorrStr.setText(self.config['spec_corr_file'])
        if self.config['isTOA5'] == 'True':
            self.isTOA5Radio.setChecked(True)
        else:
            self.isTOA5Radio.setChecked(False)
        if self.config['show_lag'] == 'True':
            self.showLag.setChecked(True)
        else:
            self.showLag.setChecked(False)
        if self.config['isLI7500'] == 'True':
            self.isLI7500Radio.setChecked(True)
        else:
            self.isLI7500Radio.setChecked(False)
        self.lagLimStr.setText(str(self.config['lag_lim']))
        self.startTimeStr.setText(self.config['t_start'])
        self.endTimeStr.setText(self.config['t_end'])
        self.sonicTimeOffsetStr.setText(str(self.config['sonic_time_offset']))
        self.sonicDirectStr.setText(self.config['sonic_head_direction'])
        idx = self.QCLGasCombo.findText(self.config['target_gas'])
        if idx > 0:
            self.QCLGasCombo.setCurrentIndex(idx)
        self.setFileStr.setText(setFile)
        self.showAdvancedSetDialog()
        self.QCLThold.setText(self.config['QCLThold'])
        idx = self.despikeMethodCombo.findText(self.config['despike_method'])
        self.despikeMethodCombo.setCurrentIndex(idx)
        self.alphaVStr.setText(str(self.config['alpha_v']))
        self.caliPresStr.setText(str(self.config['p_cal']))
        self.caliTempStr.setText(str(self.config['T_cal']))
        self.QCLPeakLoc.setText(self.config['QCLPeakLoc'])
        self.QCLPeakThold.setText(self.config['QCLPeakThold'])
        self.dStr.setText(str(self.config['d']))
        self.zStr.setText(str(self.config['z']))
        self.caliOffsetText.setText(str(self.config['caliOffset']))
        self.caliSlopeText.setText(str(self.config['caliSlope']))
        idx = self.QCLGasCombo.findText(self.config['target_gas'])
        self.QCLGasCombo.setCurrentIndex(idx)

    def getRawDir(self):
        rawDir = self.getFileDirDialog()
        self.rawDirStr.setText(rawDir)

    def getSaveDir(self):
        saveDir = self.getFileDirDialog()
        self.saveDir.setText(saveDir)

    def getStartTime(self):
        self.calDialog = QtGui.QDialog(self)
        self.calDialog.setGeometry(300, 300, 275, 175)
        self.calDialog.setWindowTitle('Calendar')
        cal = QtGui.QCalendarWidget(self.calDialog)
        cal.clicked[QtCore.QDate].connect(self.setStartTime)
        self.calDialog.show()
        self.calDialog.lbl = QtGui.QLabel(self.calDialog)
        date = cal.selectedDate()
        self.calDialog.lbl.setText(date.toString())

    def setStartTime(self, date):
        self.startTimeStr.setText(date.toString('yyyy-MM-dd') + 'T00:00')
        self.calDialog.hide()

    def getEndTime(self):
        self.calDialog = QtGui.QDialog(self)
        self.calDialog.setWindowTitle('Calendar')
        cal = QtGui.QCalendarWidget(self.calDialog)
        cal.clicked[QtCore.QDate].connect(self.setEndTime)
        print cal.geometry()
        self.calDialog.setGeometry(300, 300, 275, 175)
        self.calDialog.show()
        self.calDialog.lbl = QtGui.QLabel(self.calDialog)
        date = cal.selectedDate()
        self.calDialog.lbl.setText(date.toString())

    def setEndTime(self, date):
        self.endTimeStr.setText(date.toString('yyyy-MM-dd') + 'T00:00')
        self.calDialog.hide()

    def getFileNameDiaglog(self):
        fileName = QtGui.QFileDialog.getOpenFileName(self)
        return fileName

    def getFileDirDialog(self):
        dirName = QtGui.QFileDialog.getExistingDirectory(self)
        return dirName

    def saveConfig(self):

        self.config['col_names'] = str(self.colNameStr.text()).split(', ')
        self.config['TOA5_name'] = str(self.TOA5Str.text())
        self.config['dir'] = str(self.rawDirStr.text())
        self.config['despike_lim'] = map(float,
                                         str(self.despikeLimStr.text()).split(', '))
        self.config['file_type'] = str(self.fileTypeStr.text())
        self.config['base_time'] = str(self.basetimeStr.text())
        self.config['isTOA5'] = str(self.isTOA5Radio.isChecked())
        self.config['spec_corr_file'] = str(self.specCorrStr.text())
        self.config['t_start'] = str(self.startTimeStr.text())
        self.config['t_end'] = str(self.endTimeStr.text())
        self.config['p_cal'] = float(self.caliPresStr.text())
        self.config['T_cal'] = float(self.caliTempStr.text())
        self.config['alpha_v'] = float(self.alphaVStr.text())
        self.config['d'] = float(self.dStr.text())
        self.config['z'] = float(self.zStr.text())
        self.config['show_lag'] = str(self.showLag.isChecked())
        self.config['lag_lim'] = int(self.lagLimStr.text())
        self.config['sonic_time_offset'] = float(str(self.sonicTimeOffsetStr.text()))
        self.config['save_dir'] = str(self.saveDir.text())
        self.config['sonic_head_direction'] = str(self.sonicDirectStr.text())
        self.config['isLI7500'] = str(self.isLI7500Radio.isChecked())
        self.config['QCLThold'] = str(self.QCLThold.text())
        self.config['despike_method'] = str(self.despikeMethodCombo.currentText())
        self.config['QCLPeakLoc'] = str(self.QCLPeakLoc.text())
        self.config['QCLPeakThold'] = str(self.QCLPeakThold.text())
        self.config['prefix'] = str(self.prefixStr.text())
        self.config['caliOffset'] = str(self.caliOffsetText.text())
        self.config['caliSlope'] = str(self.caliSlopeText.text())

        print self.config['isTOA5']
        ECTools.save_config(self.config, str(self.setFileStr.text()))

    def showAdvancedSetDialog(self):
        self.advancedSetDialog.show()
        self.advancedSetDialog.activateWindow()

    def createAdvancedSettingDialog(self):
        self.radioButtonGroup = QtGui.QButtonGroup()

        self.advancedSetDialog = QtGui.QDialog()

        setVBox = QtGui.QVBoxLayout()
        setVBox.setAlignment(QtCore.Qt.AlignTop)

        # Get setting files
        setFileLabel = QtGui.QLabel('Setting file:')
        setFileHBox = QtGui.QHBoxLayout()
        self.setFileStr = QtGui.QLineEdit()
        self.setFileStr.setText(settingFile)
        setFileButton = QtGui.QPushButton('Load')
        setFileButton.clicked.connect(self.getSetFile)
        setFileHBox.addWidget(setFileLabel)
        setFileHBox.addWidget(self.setFileStr)
        setFileHBox.addWidget(setFileButton)

        # Using TOA5 or not
        isTOA5Label = QtGui.QLabel('Use TOA5 file?')
        self.isTOA5Radio = QtGui.QRadioButton()
        isTOA5HBox = QtGui.QHBoxLayout()
        isTOA5HBox.addWidget(isTOA5Label)
        isTOA5HBox.addWidget(self.isTOA5Radio)
        isTOA5HBox.setAlignment(QtCore.Qt.AlignLeft)

        # Have LI7500 or not
        isLI7500Label = QtGui.QLabel('Have LI7500?')
        self.isLI7500Radio = QtGui.QRadioButton()
        isLIHBox = QtGui.QHBoxLayout()
        isLIHBox.addWidget(isLI7500Label)
        isLIHBox.addWidget(self.isLI7500Radio)

        self.radioButtonGroup.addButton(self.isTOA5Radio)
        self.radioButtonGroup.addButton(self.isLI7500Radio)
        self.radioButtonGroup.setExclusive(False)

        # Set Tower height and displacement height
        zLabel = QtGui.QLabel('Instrument height (m): ')
        self.zStr = QtGui.QLineEdit()
        dLabel = QtGui.QLabel('Displacement height (m): ')
        self.dStr = QtGui.QLineEdit()
        towerHBox = QtGui.QHBoxLayout()
        towerHBox.addWidget(zLabel)
        towerHBox.addWidget(self.zStr)
        towerHBox.addWidget(dLabel)
        towerHBox.addWidget(self.dStr)

        # Set sonic direction
        sonicDLabel = QtGui.QLabel('Sonid hear direction: ')
        self.sonicDirectStr = QtGui.QLineEdit()
        sonicDHBox = QtGui.QHBoxLayout()
        sonicDHBox.addWidget(sonicDLabel)
        sonicDHBox.addWidget(self.sonicDirectStr)

        # Sensor raw file prefix
        prefixLabel = QtGui.QLabel('QCL file prefix:')
        prefixHBox = QtGui.QHBoxLayout()
        self.prefixStr = QtGui.QLineEdit()
        prefixHBox.addWidget(prefixLabel)
        prefixHBox.addWidget(self.prefixStr)
        okButton = QtGui.QPushButton('Ok')
        okButton.clicked.connect(self.advancedSetDialog.hide)

        # Sensor raw data file type
        fileTypeLabel = QtGui.QLabel('QCL file type:')
        self.fileTypeStr = QtGui.QLineEdit()
        prefixHBox.addWidget(fileTypeLabel)
        prefixHBox.addWidget(self.fileTypeStr)

        # Calibration pressure and temperature
        calibCondLabel = QtGui.QLabel('QCL calibration conditions:')
        caliTempLabel = QtGui.QLabel('T (K):')
        caliPresLabel = QtGui.QLabel('p (hPa):')
        self.caliTempStr = QtGui.QLineEdit()
        self.caliPresStr = QtGui.QLineEdit()
        caliHBox = QtGui.QHBoxLayout()
        caliHBox.addWidget(caliTempLabel)
        caliHBox.addWidget(self.caliTempStr)
        caliHBox.addWidget(caliPresLabel)
        caliHBox.addWidget(self.caliPresStr)

        # Water vapor effective broadening coefficient
        alphaVLabel = QtGui.QLabel('Water vapor effective broadening coeff: ')
        self.alphaVStr = QtGui.QLineEdit()
        alphaVHBox = QtGui.QHBoxLayout()
        alphaVHBox.addWidget(alphaVLabel)
        alphaVHBox.addWidget(self.alphaVStr)

        # LabView basetime
        basetimeLabel = QtGui.QLabel('Basetime in LabView: ')
        self.basetimeStr = QtGui.QLineEdit()

        # QCL intensity threshold
        QCLTholdLabel = QtGui.QLabel('QCL intensity threshold: ')
        QCLPeakLocLabel = QtGui.QLabel('QCL peak location and threshold:')
        self.QCLThold = QtGui.QLineEdit()
        self.QCLPeakLoc = QtGui.QLineEdit()
        self.QCLPeakThold = QtGui.QLineEdit()

        QCLTHoldHBox = QtGui.QHBoxLayout()
        QCLTHoldHBox.addWidget(QCLTholdLabel)
        QCLTHoldHBox.addWidget(self.QCLThold)
        QCLTHoldHBox.addWidget(QCLPeakLocLabel)
        QCLTHoldHBox.addWidget(self.QCLPeakLoc)
        QCLTHoldHBox.addWidget(self.QCLPeakThold)

        # Sonic time offset
        sonicTimeOffsetLabel = QtGui.QLabel('Sonic time offset (s): ')
        self.sonicTimeOffsetStr = QtGui.QLineEdit()
        sonicTimeHBox = QtGui.QHBoxLayout()
        sonicTimeHBox.addWidget(sonicTimeOffsetLabel)
        sonicTimeHBox.addWidget(self.sonicTimeOffsetStr)

        # Column names for input files
        colNameLabel = QtGui.QLabel('Column names:')
        self.colNameStr = QtGui.QLineEdit()

        # Despike method
        despikeMethodLabel = QtGui.QLabel('Despike method: ')
        self.despikeMethodCombo = QtGui.QComboBox()
        self.despikeMethodCombo.addItems(['constant', 'standard'])
        despikeMethodHBox = QtGui.QHBoxLayout()
        despikeMethodHBox.addWidget(despikeMethodLabel)
        despikeMethodHBox.addWidget(self.despikeMethodCombo)

        # Despike lim
        despikeLimLabel = QtGui.QLabel('Despike limits (7 int):')
        self.despikeLimStr = QtGui.QLineEdit()

        # Get calibration parameters
        caliOffsetLabel = QtGui.QLabel('Calibration Offset:')
        caliSlopeLabel = QtGui.QLabel('Slope')
        self.caliOffsetText = QtGui.QLineEdit()
        self.caliSlopeText = QtGui.QLineEdit()
        caliParamsHBox = QtGui.QHBoxLayout()
        caliParamsHBox.addWidget(caliOffsetLabel)
        caliParamsHBox.addWidget(self.caliOffsetText)
        caliParamsHBox.addWidget(caliSlopeLabel)
        caliParamsHBox.addWidget(self.caliSlopeText)

        # Get spectroscopic correction file location:
        specCorrLabel = QtGui.QLabel('Spectroscopic correction file:')
        self.specCorrStr = QtGui.QLineEdit()
        specCorrButton = QtGui.QPushButton('Load')
        specCorrButton.clicked.connect(self.getSpecCorrFile)
        specCorrHBox = QtGui.QHBoxLayout()
        specCorrHBox.addWidget(self.specCorrStr)
        specCorrHBox.addWidget(specCorrButton)

        setVBox.addLayout(setFileHBox)
        setVBox.addLayout(isTOA5HBox)
        setVBox.addLayout(isLIHBox)
        setVBox.addLayout(towerHBox)
        setVBox.addLayout(sonicDHBox)
        setVBox.addLayout(prefixHBox)
        setVBox.addWidget(calibCondLabel)
        setVBox.addLayout(caliParamsHBox)
        setVBox.addLayout(caliHBox)
        setVBox.addLayout(alphaVHBox)
        setVBox.addWidget(basetimeLabel)
        setVBox.addWidget(self.basetimeStr)
        setVBox.addLayout(QCLTHoldHBox)
        setVBox.addLayout(sonicTimeHBox)
        setVBox.addWidget(colNameLabel)
        setVBox.addWidget(self.colNameStr)
        setVBox.addWidget(despikeLimLabel)
        setVBox.addWidget(self.despikeLimStr)
        setVBox.addLayout(despikeMethodHBox)
        setVBox.addWidget(specCorrLabel)
        setVBox.addLayout(specCorrHBox)
        setVBox.addWidget(okButton)

        # Read default settings
        self.readSettingFromFile(settingFile)

        self.advancedSetDialog.setLayout(setVBox)
        self.advancedSetDialog.setSizePolicy(QtGui.QSizePolicy.Expanding,
                                             QtGui.QSizePolicy.Expanding)
        self.advancedSetDialog.hide()

    def calc_flux(self):
        # Save setting first to make sure settings are stored

        self.started = True
        print self.started

        self.saveConfig()

        if self.config['file_type'] == 'csv':
            delimiter = ','
        else:
            delimiter = '\t'

        # Get target gas str
        target_gas = str(self.QCLGasCombo.currentText())

        # Set time interval
        t_intv = pd.to_datetime([str(self.startTimeStr.text()),
                                 str(self.endTimeStr.text())])

        self.statusBox.setText('Start calculation:\n')
        self.statusBox.moveCursor(QtGui.QTextCursor.End)
        QtGui.QApplication.processEvents()

        print self.isTOA5Radio.isChecked()

        if not self.started:
            self.statusBox.insertPlainText('Calculation stopped by user.\n')
            return

        if self.isTOA5Radio.isChecked():
            # Read TOA file
            self.statusBox.insertPlainText(
                'Reading TOA5 file...')
            QtGui.QApplication.processEvents()

            TOA5_data = ECTools.read_TOA5(str(self.TOA5Str.text()), time_zone=0)

            TOA5_data['Ts'] = TOA5_data['Ts'] + 273.15  # Temperature
            TOA5_data['CO2'] = TOA5_data['CO2'] / 44  # Convert mg/m^3 to mmol/m^3
            TOA5_data['H2O'] = TOA5_data[
                                   'H2O'] / 18 * 1e3  # Convert g/m^3 to mmol/m^3
            self.statusBox.insertPlainText('...Done\n')
            QtGui.QApplication.processEvents()

        # Read raw N2O files
        self.statusBox.insertPlainText(
            'Reading ' + target_gas + ' raw files...')
        QtGui.QApplication.processEvents()
        if not self.started:
            self.statusBox.insertPlainText('Calculation stopped by user.\n')
            return

        files, date_time = ECTools.get_files(str(self.rawDirStr.text()),
                                             str(self.prefixStr.text()),
                                             t_intv,
                                             date_fmt='(\d+-\d+-\d+-T\d+)\.'
                                                      + self.config['file_type'])

        col_names = self.config['col_names']
        base_time = self.config['base_time']

        raw_data = ECTools.read_data_pd(files, col_names, base_time,
                                        delimiter=delimiter, time_zone=-4)

        # Dealing with no LI7500 case.
        if not self.isLI7500Radio.isChecked():
            raw_data['H2O'] = 0.
            raw_data['CO2'] = 0.
            raw_data['p'] = 74.

        self.statusBox.insertPlainText('...Synchronizing...')
        QtGui.QApplication.processEvents()
        if not self.started:
            self.statusBox.insertPlainText('Calculation stopped by user.\n')
            return

        raw_data = ECTools.time_realign(raw_data)
        raw_data[target_gas] = raw_data['2f_peak'] / raw_data[
            'intensity'] * float(self.config['caliSlope']) + float(
            self.config['caliOffset'])

        if target_gas == 'N2O':
            # Calculate N2O concentration
            invalidMask = ECTools.QC_QCL_NH3(raw_data, ['peak_loc', 'intensity'],
                                             6500, 30,
                                             1.0, t_window=0.5)
            raw_data[target_gas][invalidMask] = np.nan
            t_delta = np.timedelta64(8, unit='h').astype('timedelta64[ns]')
            # Combine the two data sets
            data = raw_data.combine_first(TOA5_data[t_intv[0] - t_delta:t_intv[
                                                                            1] - t_delta])
            data.loc[data['agc'] > 50, :] = np.nan
        else:
            raw_data['Ts'] = ECTools.calc_sonic_T(raw_data['c'])
            invalid_mask = ECTools.QC_QCL_NH3(raw_data, ['peak_idx',
                                                         'intensity'],
                                              float(self.config['QCLPeakLoc']),
                                              float(self.config['QCLPeakThold']),
                                              float(self.config['QCLThold']))
            raw_data['intensity'][invalid_mask] = np.nan
            raw_data['NH3'][invalid_mask] = np.nan
            data = raw_data

        self.statusBox.insertPlainText('...Done\n')
        QtGui.QApplication.processEvents()
        if not self.started:
            self.statusBox.insertPlainText('Calculation stopped by user.\n')
            return

        # Resample to 10 sec data for plotting
        self.statusBox.insertPlainText('Plotting data...')
        QtGui.QApplication.processEvents()

        ECTools.despike(data, [target_gas, 'H2O', 'CO2', 'u', 'v', 'w', 'Ts', 'p',
                               'intensity'],
                        self.config['despike_lim'],
                        method=self.config['despike_method'], n=30)
        if target_gas == 'N2O':
            ECTools.shift_sonic_time(data, ['u', 'v', 'w', 'Ts', 'CO2', 'H2O',
                                            'p', 'agc'],
                                     self.config['sonic_time_offset'])
        else:
            ECTools.shift_sonic_time(data, ['u', 'v', 'w', 'Ts'],
                                     self.config['sonic_time_offset'])

        data[target_gas][np.isnan(data['intensity'])] = np.nan

        if not self.config['isTOA5']:
            data['CO2'][np.isnan(data['intensity'])] = np.nan
            data['H2O'][np.isnan(data['intensity'])] = np.nan

        avg_data = data[::100]

        # Plot raw data
        ECTools.plot_time_series_pd(avg_data, ['Ts', 'p', 'H2O', 'CO2', 'intensity',
                                               target_gas], create_fig=False,
                                    fig=self.canvasTS.fig,
                                    axes=self.canvasTS.axes, labels=['T (K)',
                                                                     'p (kPa)',
                                                                     'H2O ('
                                                                     'mmol/m$^3$)',
                                                                     'CO2 ('
                                                                     'mmol/m$^3)$',
                                                                     'intensity ('
                                                                     'V)',
                                                                     target_gas + ' '
                                                                                  '(ppbv)'])
        self.statusBox.insertPlainText('...Done\n')
        QtGui.QApplication.processEvents()
        if not self.started:
            self.statusBox.insertPlainText('Calculation stopped by user.\n')
            return

        # Load spectroscopic correction factors
        cf_file = self.config['spec_corr_file']
        df_cf = pd.read_csv(cf_file)

        block_data_out = ECTools.create_output_data(target_gas)
        cospectra_out = []

        count = data.resample('30min').apply(ECTools.count_non_nan)
        t_delta = pd.to_timedelta('30min')

        # Get interp2d object for spectroscopic correction.
        spec_correction = LinearNDInterpolator(
            np.c_[df_cf['p_grid'], df_cf['t_grid']], df_cf['correction_factor'])

        lag_lim = int(self.lagLimStr.text())

        for t_idx in count.index:
            self.statusBox.insertPlainText(
                'Calculating ' + t_idx.strftime('%Y-%m-%d %H:%M') + '...')
            QtGui.QApplication.processEvents()
            if not self.started:
                self.statusBox.insertPlainText('Calculation stopped by user.\n')
                return

            if sum(count.loc[
                       t_idx, [target_gas, 'CO2', 'H2O', 'w', 'Ts']] > 9000) == 5:

                tmp_data = data[t_idx: t_idx + t_delta]

                ECTools.shift_lag(tmp_data, [target_gas, 'H2O', 'CO2'], 'w',
                                  lag_lim=self.config['lag_lim'],
                                  show_fig=self.showLag.isChecked(),
                                  create_fig=False, fig=self.canvasXCov.fig,
                                  axes=self.canvasXCov.axes)
                QtGui.QApplication.processEvents()

                tur_flux, air_prop, gas_den_avg, cospectra, sst, \
                itc = ECTools.calc_turflux_30min(tmp_data, t_idx, target_gas,
                                                 self.config['d'],
                                                 self.config['z'],
                                                 self.config['alpha_v'],
                                                 self.config['p_cal'],
                                                 self.config['T_cal'],
                                                 spec_correction)

                tmp_data[0:18000].to_csv(self.config['save_dir'] + '\\' +
                                         target_gas +
                                         '_30min_' + t_idx.strftime('%Y%m%dT%H%M')
                                         + '.csv', na_rep='NaN')

                print tur_flux

            else:
                tur_flux, air_prop, gas_den_avg, cospectra, sst, \
                itc = ECTools.fill_null(target_gas)

            ECTools.combine_output_data(block_data_out, target_gas,
                                        t_idx.to_datetime(),
                                        air_prop,
                                        gas_den_avg, tur_flux, sst, itc)
            cospectra_out.append(cospectra)

            self.statusBox.insertPlainText('...Done\n')
            self.statusBox.moveCursor(QtGui.QTextCursor.End)
            QtGui.QApplication.processEvents()

        df_block_data_out = pd.DataFrame(block_data_out)
        df_block_data_out['Time'] = pd.to_datetime(df_block_data_out['Time'])
        df_block_data_out.set_index('Time', drop=False, inplace=True)
        df_block_data_out['WPL_' + target_gas] = ECTools.molec_weight[
                                                     target_gas] * \
                                                 df_block_data_out['w_WPL'] * \
                                                 df_block_data_out[target_gas +
                                                                   '_avg']
        df_block_data_out['x_' + target_gas] = df_block_data_out[
                                                   target_gas + '_avg'] * \
                                               8.31 * df_block_data_out['T_avg'] / \
                                               df_block_data_out['p_avg'] * 1e6

        df_block_data_out['wind_direction'] += float(self.config[
                                                         'sonic_head_direction'])

        df_block_data_out['wind_direction'] %= 360.

        ECTools.plot_time_series_pd(df_block_data_out, ['H', 'LE', 'T_avg',
                                                        'x_' + target_gas, 'F_' +
                                                        target_gas,
                                                        'WPL_' + target_gas],
                                    create_fig=False,
                                    fig=self.canvasEC.fig,
                                    axes=self.canvasEC.axes)

        df_block_data_out.to_csv(self.config['save_dir'] + '\\' + target_gas +
                                 '_results_' +
                                 self.config['t_end'].replace(':', '') + '.csv')

        pl.dump(cospectra_out, open(target_gas + '_cosp_' + \
                                    self.config['t_end'].replace(':', '') + \
                                    '.pl', 'wb'))

        self.statusBox.insertPlainText('Calculation finished.')
        QtGui.QApplication.processEvents()

    def stop_calc(self):
        if self.started:
            self.started = False


def resource_path(relative):
    if hasattr(sys, "_MEIPASS"):
        return os.path.join(sys._MEIPASS, relative)
    return os.path.join(relative)


def main():
    app = QtGui.QApplication(sys.argv)
    appWindow = AppWindow()
    appWindow.show()
    sys.exit(app.exec_())


filename = 'defaultSettings.txt'
cwd = os.getcwd()
settingFile = resource_path(os.path.join(cwd, filename))

if __name__ == '__main__':
    main()
