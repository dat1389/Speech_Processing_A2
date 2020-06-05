from PyQt5.QtWidgets import QVBoxLayout, QLabel, QPushButton, QMainWindow, QApplication, QWidget, QFileDialog
from PyQt5.QtCore import *
from PyQt5.QtGui import *

import sys
import os, queue, multiprocessing
import librosa
from test_soundfile import runHMM

class SpeechApp(QMainWindow):
    def __init__(self):
        super(SpeechApp, self).__init__()
        self.setWindowTitle("SP-APP")
        self.setFixedSize(400,300)

        self.RECORDING = False
        
        self.recordingThread = None
        self.q = queue.Queue()

        self.currentFile = None

        layout = QVBoxLayout()

        self.resultLabel = QLabel("Conclusion: ")
        self.start_record_button = QPushButton("Start Recording")
        self.stop_record_button = QPushButton("Stop Recording")
        self.choose_file_button = QPushButton("Choose file (.wav)")
        self.check_word_button = QPushButton("Check")
        self.currentFileLabel = QLabel("Current file: None")
        self.start_record_button.setStyleSheet("background-color: lightblue;"
                                               "border-style: outset;"
                                               "border-width: 2px;"
                                               "border-radius: 10px;"
                                               "border-color: beige;"
                                               "font: bold 14px;"
                                               "min-width: 10em;"
                                               "padding: 6px;")
        self.stop_record_button.setStyleSheet("background-color: red;"
                                               "border-style: outset;"
                                               "border-width: 2px;"
                                               "border-radius: 10px;"
                                               "border-color: beige;"
                                               "font: bold 14px;"
                                               "min-width: 10em;"
                                               "padding: 6px;")
        self.choose_file_button.setStyleSheet("background-color: green;"
                                               "color: white;"
                                               "border-style: outset;"
                                               "border-width: 2px;"
                                               "border-radius: 10px;"
                                               "border-color: beige;"
                                               "font: bold 14px;"
                                               "min-width: 10em;"
                                               "padding: 6px;") 
        self.check_word_button.setStyleSheet("background-color: red;"
                                               "border-style: outset;"
                                               "border-width: 2px;"
                                               "border-radius: 10px;"
                                               "border-color: beige;"
                                               "font: bold 14px;"
                                               "min-width: 10em;"
                                               "padding: 6px;")                                     
        self.start_record_button.clicked.connect(self.startRecord)
        self.stop_record_button.clicked.connect(self.stopRecord)
        self.choose_file_button.clicked.connect(self.chooseFile)
        self.check_word_button.clicked.connect(self.checkWord)
        
        layout.addWidget(self.start_record_button)
        layout.addWidget(self.stop_record_button)
        layout.addWidget(self.choose_file_button)
        layout.addWidget(self.currentFileLabel)
        layout.addWidget(self.resultLabel)
        layout.addWidget(self.check_word_button)

        widget = QWidget()
        widget.setLayout(layout)

        self.setCentralWidget(widget)

    def showAnswer(self, answer):
        self.resultLabel.setText("Conclusion: " + answer)

    def startRecord(self):
        #print("Start recording")
        #os.remove("live_recording.wav")
        self.RECORDING = True
        self.runRecording()

    def stopRecord(self):
        self.RECORDING = False
        #print("Stop recording")
        self.recordingThread = None
        self.changeFile("live_recording.wav")

    def chooseFile(self):
        soundfile, _ = QFileDialog.getOpenFileName(self, "Open sound", "", "soundile (*wav)")
        soundfile = soundfile.rstrip(os.sep)
        self.changeFile(soundfile)
        
    def changeFile(self, file_path):
        self.currentFile = file_path
        self.currentFileLabel.setText("Current file: " + file_path)

    def checkWord(self):
        if (self.currentFile is None):
            print("No file specified.")
            return

        evals, conclusion = runHMM(self.currentFile)

        print(evals)
        self.showAnswer(conclusion)

    # Recording functions
    SAMPLE_RATE = 22050
    CHANNELS = 1

    def runRecording(self):
        if (os.path.isfile("live_recording.wav")):
            os.remove("live_recording.wav")

        self.recordingThread = multiprocessing.Process(target=self.record, args=(lambda: not self.RECORDING,), daemon=True)
        self.recordingThread.start()
            
    def callback(self, indata, frames, time, status):
        #This is called (from a separate thread) for each audio block.
        if status:
            print(status, file=sys.stderr)
        self.q.put(indata.copy())

    def record(self, stop):
        import sounddevice as sd
        import soundfile as sf

        try:
            #Open a new soundfile and attempt recording
            with sf.SoundFile("live_recording.wav", mode='x', samplerate=self.SAMPLE_RATE, channels=1, subtype="PCM_24") as file:
                with sd.InputStream(samplerate=self.SAMPLE_RATE, device=sd.default.device, channels=1, callback=self.callback):                
                    while True:
                        file.write(self.q.get())
                        if stop():
                            break

                    print("Stopped")

        except Exception as e:
            print(e)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    gui = SpeechApp()
    gui.show()
    app.exec_()