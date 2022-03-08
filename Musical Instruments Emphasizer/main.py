from operator import index
from re import X
import sounddevice as sd
from PyQt5 import QtWidgets
from PyQt5.QtCore import QUrl
from PyQt5.QtWidgets import *
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from pyqtgraph import *
from pyqtgraph import PlotWidget, PlotItem
import pyqtgraph as pg
import pandas as pd
import matplotlib.pyplot as plt
import sys
import os
import pathlib
import numpy as np
from pyqtgraph.Qt import _StringIO
from PyQt5.QtWidgets import QApplication, QWidget, QInputDialog, QLineEdit, QFileDialog
from PyQt5.QtGui import QIcon
from matplotlib.figure import Figure
import pyqtgraph.exporters
import math
import winsound
from PyQt5.QtMultimedia import QMediaContent, QMediaPlayer
from scipy.io import wavfile
from finalgui import Ui_MainWindow, MplCanvas
import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import rfft, rfftfreq, irfft
import cmath
from scipy.io.wavfile import write

class GuitarString:
    def __init__(self, pitch, starting_sample, sampling_freq, stretch_factor):
        self.the_pitch = pitch
        self.the_starting_sample = starting_sample
        self.the_sampling_freq = sampling_freq
        self.stretch_factor = stretch_factor
        self.init_the_wavetable()
        self.current_sample = 0
        self.previous_value = 0

    def init_the_wavetable(self):
        piano_wavetable_size = self.the_sampling_freq // int(self.the_pitch)
        self.guitar_wavetable = (2 * np.random.randint(0, 2, piano_wavetable_size) - 1).astype(float)

    def get_the_sample(self):
        if self.current_sample >= self.the_starting_sample:
            current_sample_mod = self.current_sample % self.guitar_wavetable.size
            drawn_samples = np.random.binomial(1, 1 - 1 / self.stretch_factor)
            if drawn_samples == 0:
                self.guitar_wavetable[current_sample_mod] = 0.5 * (
                        self.guitar_wavetable[current_sample_mod] + self.previous_value)
            sample = self.guitar_wavetable[current_sample_mod]
            self.previous_value = sample
            self.current_sample += 1
        else:
            self.current_sample += 1
            sample = 0
        return sample

class MainWindow(QtWidgets.QMainWindow):
    gains = np.array([1.0, 1.0, 1.0, 1.0, 1.0])  # to store the gain of each slider
    levels_min = [0, 500, 2000, 1000, 5000]
    levels_max = [500, 1000, 5000, 2000, 10000]
    the_data_of_song = []
    the_sample_rate_of_drum = 11000
    the_sample_rate_of_guitar = 8000
    the_samplerate_of_piano = 44100  # Hz
    wavetable_size = 200
    isPaused = False
    drums_keys = ["DR1", "DR2"]
    piano_Keys = ['W', "w", 'Q', 'q', 'Y', 'H', 'L',
                  'O', 'o', 'T', 't', 'X', "E", "e",
                  "u", 'R', "r", 'G', 'U', 'u', 'I', 'i', 'Weq', 'OiL', 'Wqw', 'tEi', 'rHu', 'GuT', 'YwX', 'tRi', 'IoY',
                  'LrU', 'uTe', 'LiY', 'TEt', 'eWi', 'HqT'
                  ]

    guitar_strings = ["GUITAR_1", "GUITAR_2", "GUITAR_3", "GUITAR_4", "GUITAR_5"]
    the_base_freq = 261.63
    freqs = [98, 123, 147, 196, 240]
    the_unit_delay = 1000
    Move_to_the_right = 0
    The_step = 0
    def __init__(self, *args, **kwargs):
        super(MainWindow, self).__init__()
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)

        self.the_note_freqs = {self.piano_Keys[element_num]: self.the_base_freq * pow(2, (element_num / 37))
                           for element_num in range(37)}

        #______________________________IMPORTANT INSTANCE VARIABLES__________________________________________________________#
        self.the_delays = [self.the_unit_delay * frequency for frequency in range(len(self.freqs))]
        self.the_stretch_factors = [2 * f / 98 for f in self.freqs]
        self.timer1 = QtCore.QTimer()
        self.data_lines = []
        self.startTimeIdx = 0
        self.counter=0
        self.zoomFactor = 1
        self.Timer = [self.timer1]
        self.GraphicsView=[self.ui.graphicsView_3]#ALL GRAPHIGSVIEW TO USE THEM WIH JUST INDEX
        self.white=mkPen(color=(255,255, 255))#white
        self.Color1=mkPen(color=(255, 0, 0))#RED
        self.Color2=mkPen(color=(0, 255, 0))#GREEN
        self.Color3=mkPen(color=(0, 0, 255))#BLUE
        self.Color4=mkPen(color=(255, 200, 200), style=QtCore.Qt.DotLine)#Dotted pale-red line
        self.Color5=mkPen(color=(200, 255, 200), style=QtCore.Qt.DotLine)#Dotted pale-green line
        self.Color6=mkPen(color=(200, 200, 255), style=QtCore.Qt.DotLine)## Dotted pale-blue line
        self.COLOR_Pen=[self.Color1,self.Color2,self.Color3,self.Color4,self.Color5,self.Color6]#STORE COLORS TO BE USED WITH INDEX
        self.music_player = QMediaPlayer()
        self.The_Sliders = [self.ui.PIANO_SLIDER, self.ui.VIOLIN_SLIDER, self.ui.DRUMS_SLIDER, self.ui.SAX_SLIDER, self.ui.GUITAR_SLIDER]
        #self.slidervalues = np.ones((10,), dtype=float)
        for x in range(len(self.The_Sliders)):
            self.connect_sliders(x)
            self.The_Sliders[x].setMaximum(100)
            self.The_Sliders[x].setMinimum(0)
            self.The_Sliders[x].setValue(100)
        #___________________________________________CONNECTING BUTTONS WITH THEIR FUNCTIONS_______________________________________#
        self.ui.Open.triggered.connect(lambda: self.open())
        self.ui.Clear.triggered.connect(lambda: self.clear())
        self.ui.VOLUME_SLIDER.valueChanged.connect(lambda: self.change_Volume())
        self.ui.PLAY.clicked.connect(lambda: self.play_msc())
        self.th = {}
        #-----------------------------------------------------------------------#
        self.ui.PIANO.clicked.connect(lambda: self.PIANO_PAGE())
        self.ui.DRUMS.clicked.connect(lambda: self.DRUMS_PAGE())
        self.ui.GUITAR.clicked.connect(lambda: self.GUITAR_PAGE())
        #---------------------------------------------------------------#
        self.ui.PIANO_W_7.clicked.connect(lambda: self.generate())
        self.ui.PIANO_W_8.clicked.connect(lambda: self.generate())
        self.ui.PIANO_W_9.clicked.connect(lambda: self.generate())
        self.ui.PIANO_W_10.clicked.connect(lambda: self.generate())
        self.ui.PIANO_W_11.clicked.connect(lambda: self.generate())
        self.ui.PIANO_W_12.clicked.connect(lambda: self.generate())
        self.ui.PIANO_W_13.clicked.connect(lambda: self.generate())
        self.ui.PIANO_W_14.clicked.connect(lambda: self.generate())

        self.ui.PIANO_W_15.clicked.connect(lambda: self.generate())
        self.ui.PIANO_W_16.clicked.connect(lambda: self.generate())
        self.ui.PIANO_W_17.clicked.connect(lambda: self.generate())
        self.ui.PIANO_W_18.clicked.connect(lambda: self.generate())
        self.ui.PIANO_W_19.clicked.connect(lambda: self.generate())
        self.ui.PIANO_W_20.clicked.connect(lambda: self.generate())
        self.ui.PIANO_W_21.clicked.connect(lambda: self.generate())
        self.ui.PIANO_W_22.clicked.connect(lambda: self.generate())

        self.ui.PIANO_B_1.clicked.connect(lambda: self.generate())
        self.ui.PIANO_B_2.clicked.connect(lambda: self.generate())
        self.ui.PIANO_B_3.clicked.connect(lambda: self.generate())
        self.ui.PIANO_B_4.clicked.connect(lambda: self.generate())
        self.ui.PIANO_B_5.clicked.connect(lambda: self.generate())
        self.ui.PIANO_B_6.clicked.connect(lambda: self.generate())
        self.ui.PIANO_B_7.clicked.connect(lambda: self.generate())
        self.ui.PIANO_B_8.clicked.connect(lambda: self.generate())
        self.ui.PIANO_B_9.clicked.connect(lambda: self.generate())
        self.ui.PIANO_B_10.clicked.connect(lambda: self.generate())

        self.ui.PIANO_B_11.clicked.connect(lambda: self.generate())
        self.ui.PIANO_B_12.clicked.connect(lambda: self.generate())
        self.ui.PIANO_B_13.clicked.connect(lambda: self.generate())
        self.ui.PIANO_B_14.clicked.connect(lambda: self.generate())
        self.ui.PIANO_B_15.clicked.connect(lambda: self.generate())

        self.ui.PIANO_W_6.clicked.connect(lambda: self.generate())
        self.ui.PIANO_W_1.clicked.connect(lambda: self.generate())
        self.ui.PIANO_W_2.clicked.connect(lambda: self.generate())
        self.ui.PIANO_W_3.clicked.connect(lambda: self.generate())
        self.ui.PIANO_W_4.clicked.connect(lambda: self.generate())
        self.ui.PIANO_W_5.clicked.connect(lambda: self.generate())

        #--------------------------------------------------------------#
        self.ui.GUITAR_1.clicked.connect(lambda: self.generate())
        self.ui.GUITAR_2.clicked.connect(lambda: self.generate())
        self.ui.GUITAR_3.clicked.connect(lambda: self.generate())
        self.ui.GUITAR_4.clicked.connect(lambda: self.generate())
        self.ui.GUITAR_5.clicked.connect(lambda: self.generate())
        #--------------------------------------------------------------#
        self.ui.DR1.clicked.connect(lambda: self.generate())
        self.ui.DR2.clicked.connect(lambda: self.generate())
        
    #_____________________________________________BUTTTONS FUNCTIONS_______________________________________________________# 
    def PIANO_PAGE(self):
        self.ui.stackedWidget.setCurrentWidget(self.ui.HOME_PAGE)
        
    def DRUMS_PAGE(self):
        self.ui.stackedWidget.setCurrentWidget(self.ui.DRUMS_PAGE)
        
    def GUITAR_PAGE(self):
        self.ui.stackedWidget.setCurrentWidget(self.ui.GUITAR_PAGE)
  
    #---------------------------------------------------------------------------------------------------#
    def get_the_wave(self, freq, the_duration_time=0.5):
        amplitude = 4096
        wave_on_x_axis = np.linspace(0, the_duration_time, int(self.the_samplerate_of_piano * the_duration_time))
        the_wave = amplitude * np.sin(2 * np.pi * freq * wave_on_x_axis)

        return the_wave

    def get_song_data(self, the_music_note):
        if len(self.ui.mw.sender().objectName()) > 1:
            the_genereted_tune = self.get_the_data_of_chord(self.ui.mw.sender().objectName())

        else:

            the_genereted_tune = [self.get_the_wave(self.the_note_freqs[the_music_note])]
            the_genereted_tune = np.concatenate(the_genereted_tune)

        the_genereted_tune = the_genereted_tune * (16300 / np.max(the_genereted_tune))

        return the_genereted_tune.astype(np.int16)

    def get_the_data_of_chord(self, the_chords):
        the_chords = the_chords.split('-')

        chord_data = []
        for chord in the_chords:
            the_key_data = sum([self.get_the_wave(self.the_note_freqs[piano_key]) for piano_key in list(chord)])
            chord_data.append(the_key_data)

        chord_data = np.concatenate(chord_data, axis=0)

        return chord_data.astype(np.int16)

    def get_the_sound_of_drums(self, drum_wavetable, n_samples, probability):
        samples = []
        current_sample = 0
        previous_value = 0
        while len(samples) < n_samples:
            drawn_samples = np.random.binomial(1, probability)
            sign = float(drawn_samples == 1) * 2 - 1
            drum_wavetable[current_sample] = sign * 0.5 * (drum_wavetable[current_sample] + previous_value)
            samples.append(drum_wavetable[current_sample])
            previous_value = samples[-1]
            current_sample += 1
            current_sample = current_sample % drum_wavetable.size
        return np.array(samples)

    def generate(self):
        if self.ui.mw.sender().objectName() in self.piano_Keys:
            key_or_chord_data = self.get_song_data(self.ui.mw.sender().objectName())

            sd.play(key_or_chord_data, self.the_samplerate_of_piano)

        elif self.ui.mw.sender().objectName() in self.drums_keys:
            drum_wavetable = np.ones(self.wavetable_size)
            if self.ui.mw.sender().objectName() == "DR1":
                drum_data = self.get_the_sound_of_drums(drum_wavetable, self.the_sample_rate_of_drum, 1)
            else:
                drum_data = self.get_the_sound_of_drums(drum_wavetable, self.the_sample_rate_of_drum, 0.3)

            sd.play(drum_data, self.the_sample_rate_of_drum)

        elif self.ui.mw.sender().objectName() in self.guitar_strings:

            if self.ui.mw.sender().objectName() == "GUITAR_1":
                String1 = GuitarString(self.freqs[0], self.the_delays[0], self.the_sample_rate_of_guitar,
                                            self.the_stretch_factors[0])

                guitar_sound = [String1.get_the_sample() for sample in range(self.the_sample_rate_of_guitar)]

            elif self.ui.mw.sender().objectName() == "GUITAR_2":
                String2 = GuitarString(self.freqs[1], self.the_delays[1], self.the_sample_rate_of_guitar,
                                            self.the_stretch_factors[1])

                guitar_sound = [String2.get_the_sample() for sample in range(self.the_sample_rate_of_guitar)]

            elif self.ui.mw.sender().objectName() == "GUITAR_3":

                String3 = GuitarString(self.freqs[2], self.the_delays[2], self.the_sample_rate_of_guitar,
                                            self.the_stretch_factors[2])

                guitar_sound = [String3.get_the_sample() for sample in range(self.the_sample_rate_of_guitar)]

            elif self.ui.mw.sender().objectName() == "GUITAR_4":

                String4 = GuitarString(self.freqs[3], self.the_delays[3], self.the_sample_rate_of_guitar,
                                            self.the_stretch_factors[3])

                guitar_sound = [String4.get_the_sample() for sample in range(self.the_sample_rate_of_guitar)]

            elif self.ui.mw.sender().objectName() == "GUITAR_5":

                String5 = GuitarString(self.freqs[4], self.the_delays[4], self.the_sample_rate_of_guitar,
                                            self.the_stretch_factors[4])

                guitar_sound = [String5.get_the_sample() for sample in range(self.the_sample_rate_of_guitar)]

            sd.play(guitar_sound, self.the_sample_rate_of_guitar)


    def open(self):
        files_name = QFileDialog.getOpenFileName(self, 'Open only wav', os.getenv('HOME'), "wav(*.wav)")
        self.path = files_name[0]
        print(self.path)
        full_file_path = self.path
        self.url = QUrl.fromLocalFile(full_file_path)
        if pathlib.Path(self.path).suffix == ".wav":
            self.samplerate, self.data = wavfile.read(self.path)
            self.amplitude = np.int32((self.data))
            self.time = np.linspace(0., len(self.data) / self.samplerate, len(self.data))
            self.maxAmplitude = self.amplitude.max()
            self.minAmplitude = self.amplitude.min()
            self.endTimeIdx = int(self.samplerate * self.zoomFactor) - 1
            self.content = QMediaContent(self.url)
            self.music_player.setMedia(self.content)
            self.music_player.play()
            self.plot()
            self.ui.PLAY.setText("STOP")
            self.The_Step = 0
        self.SPECTROGRAM()

    def connect_sliders(self, slider_index):
        self.The_Sliders[slider_index].sliderReleased.connect(lambda: self.Equalizer())

    # def slidervalue(self, slider_index):
    #     self.value = self.The_Sliders[slider_index].value()
    #     self.modify_signal(slider_index, self.value)

    def update_plot_data(self):  # ------------>>UPDATE THE VALUES FOR A LIVE SIGNAL<<
        self.Move_to_the_right = self.The_Step + 2
        self.The_Step += 0.11
        self.GraphicsView[0].plotItem.setXRange(self.The_Step, self.Move_to_the_right)
        if int(self.The_Step) == int(self.time[-1]):
            self.Timer[0].stop()

    def update_starter_point_and_plot(self, bol_play=True):
        self.x_mid = (self.step + self.step_right) / 2
        self.starter_point = (self.x_mid / self.duration) * len(self.song_data)
        if (bol_play):
            sd.play(self.song_data[int(self.starter_point):], self.samplerate)

    def plot(self):
        self.GraphicsView[0].clear()
        self.GraphicsView[0].setXRange(self.time[self.startTimeIdx], self.time[self.endTimeIdx])
        self.GraphicsView[0].setYRange(self.minAmplitude, self.maxAmplitude)
        self.pen = pg.mkPen(color=(121, 161, 60))
        self.GraphicsView[0].plot(self.time, self.amplitude, pen=self.pen)
        # self.GraphicsView[0].plotItem.showGrid(True, True, alpha=1)
        self.Timer[0].setInterval(100)
        if self.Move_to_the_right == 0:
            self.Timer[0].timeout.connect(lambda: self.update_plot_data())

        self.Timer[0].start()

    def SPECTROGRAM(self):#-------------------->>DRAW THE SPECTROGRAM<<
        for i in reversed(range(self.ui.verticalLayout_4.count())):
            self.ui.verticalLayout_4.itemAt(i).widget().deleteLater()
        self.ui.sc_1 = MplCanvas(self.ui.verticalLayoutWidget_4, width=5, height=5, dpi=100)
        self.ui.verticalLayout_4.addWidget(self.ui.sc_1)
        spec, freqs, t, im = self.ui.sc_1.axes.specgram(self.amplitude,Fs=self.samplerate,cmap='plasma')
        self.ui.sc_1.figure.colorbar(im).set_label('Intensity [dB]')
        self.ui.sc_1.draw()


    # def convert_to_fft(self):
    #     self.original_signal_fft = np.fft.fft(self.data)
    #     self.modified_fft = np.copy(self.original_signal_fft)  # this is the copy to operate on and update
    #     self.fft_fre = np.fft.fftfreq(n=len(self.data),
    #                                   d=1 / self.samplerate)  # gets the sample frequency bins per cycle
    #     self.freq_bins = int(len(self.data) * 0.5)  # to stop the mirroring
    #     self.update_plots(self.original_signal_fft)
        
    # def convert_ifft(self, frequency_controld_sig):
    #     bol_play = True
    #     self.song_data = np.fft.ifft(frequency_controld_sig)
    #     # self.song_data = self.song_data.real
    #     self.GraphicsView[0].clear()
    #     self.song_data = np.ascontiguousarray(self.song_data, dtype=np.int32)  # need to convert to numpy array
    #     if self.ui.PLAY.text() == "PLAY":
    #         bol_play = False
    #     self.update_starter_point_and_plot(bol_play)
    #     self.GraphicsView[0].plot(self.time, self.song_data)


    # def frequency_control(self, min_freq, max_freq, level, gain):
    #     self.modified_fft[(self.fft_fre >= min_freq) & (self.fft_fre <= max_freq)] = \
    #         self.modified_fft[(self.fft_fre >= min_freq) & (self.fft_fre <= max_freq)] / self.gains[level]
    #     self.modified_fft[(self.fft_fre <= -min_freq) & (self.fft_fre >= -max_freq)] = \
    #         self.modified_fft[(self.fft_fre <= -min_freq) & (self.fft_fre >= -max_freq)] / self.gains[level]
    #     # divide by old gain before doing new change so it wont accumulate
    #     self.gains[level] = gain
    #     self.modified_fft[(self.fft_fre >= min_freq) & (self.fft_fre <= max_freq)] = \
    #         self.modified_fft[(self.fft_fre >= min_freq) & (self.fft_fre <= max_freq)] * self.gains[level]
    #     self.modified_fft[(self.fft_fre <= -min_freq) & (self.fft_fre >= -max_freq)] = \
    #         self.modified_fft[(self.fft_fre <= -min_freq) & (self.fft_fre >= -max_freq)] * self.gains[level]

    #     #self.update_plots(self.modified_fft)
    #     self.convert_ifft(self.modified_fft)


    # def modify_signal(self, level_idx, gain):
    #     gain += 0.00001
    #     self.frequency_control(self.levels_min[level_idx], self.levels_max[level_idx], level=level_idx, gain=gain)

    def Equalizer(self):
        pianoFreqs=np.arange(80,1200)
        N = len(self.data) #*DURATION
        yf = rfft(self.data)#signal freqs Amplitudes
        xf = rfftfreq(N, 1 / self.samplerate)#signal frequences
        for freq in xf :
           if freq in pianoFreqs:
               yf[index(freq)]=yf[index(freq)]*(self.The_Sliders[0].value()/100)
        new_sig = irfft(yf)
        write("example.wav", self.samplerate, new_sig.astype(np.int16))
        self.clear()
        self.GraphicsView[0].plot( new_sig[:1000], pen=self.pen)
        # self.GraphicsView[0].plotItem.showGrid(True, True, alpha=1)
        self.Timer[0].setInterval(100)
        if self.Move_to_the_right == 0:
            self.Timer[0].timeout.connect(lambda: self.update_plot_data())
        self.Timer[0].start()
        

    def change_Volume(self):
        the_current_Volume = self.ui.VOLUME_SLIDER.value()
        self.music_player.setVolume(the_current_Volume)

    def play_msc(self):
        if self.ui.PLAY.text() == "STOP":
            self.isPaused = True
            self.music_player.pause()
            self.Timer[0].stop()
            self.ui.PLAY.setText("PLAY")
        else:
            self.isPaused = False
            self.music_player.play()
            self.Timer[0].start()
            if int(self.The_Step) == int(self.time[-1]):
                self.Timer[0].stop()
                self.music_player.setMedia(QMediaContent(None))
            self.ui.PLAY.setText("STOP")


    def clear(self):#------------------------------>>CLEAR THE 2 GRAPHS<<
        self.music_player.pause()
        self.Timer[0].stop()
        self.GraphicsView[0].clear()


#---------------------------------END OF MAINWINDOW CLASS---------------------------------------------#


if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    main = MainWindow()
    main.show()
    sys.exit(app.exec_())