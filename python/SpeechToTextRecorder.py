import tkinter
import tkinter.filedialog
import tkinter.messagebox
import getpass
# import soundcard as sc
import sounddevice as sd
import numpy as np
import pandas as pd
from queue import Queue
from datetime import datetime as dt
from pathlib import Path
from GermanSpeechToTextTranslaterBase import GermanSpeechToTextTranslaterBase
from RecordingThread import RecordingThread
from SplitterThread import SplitterThread


class SpeechRecordingGui:
    root = tkinter.Tk()

    def __init__(self, recording_base_path: str, translator: GermanSpeechToTextTranslaterBase = None):
        self.translator = translator
        self.ds_id = 'Recording'
        self.allow_recording = False
        # speakers = sc.all_speakers()
        # self.speaker = sc.default_speaker()
        # mics = sc.all_microphones()
        # self.mic = sc.default_microphone()
        self.mics = sd.query_devices(kind='input')
        self.frame_rate = 16_000
        self.frame_duration_ms = 30
        self.channels = 1
        self.frame_size = int((self.frame_rate / 1000) * self.frame_duration_ms * 2)
        self.recording_base_path = recording_base_path
        self.recording_path = f'{self.recording_base_path}/Full{self.ds_id}'
        self.snippet_path = f'{self.recording_base_path}/{self.ds_id}'
        self.file_name = f'{self.recording_path}/test.mp3'
        self.q = Queue()
        self.finished = False
        self.record_thread = RecordingThread(self.q)
        self.write_thread = SplitterThread(self.snippet_path, self.q, self.translator)

        self.initiate_gui()

    def initiate_gui(self):
        self.btn_start = tkinter.Button(SpeechRecordingGui.root, text='Start', command=self.start)

        self.btn_start.place(x=30, y=20, width=100, height=20)
        self.btn_stop = tkinter.Button(SpeechRecordingGui.root, text='Stop', command=self.stop)
        self.btn_stop.place(x=140, y=20, width=100, height=20)

        # Status of recording
        self.lb_status = tkinter.Label(SpeechRecordingGui.root, text='Ready', anchor='w', fg='green')
        self.lb_status.place(x=30, y=50, width=200, height=20)

        SpeechRecordingGui.root.title('Recorder')
        SpeechRecordingGui.root.geometry('270x80+950+300')
        SpeechRecordingGui.root.resizable(True, True)

    def start(self):
        self.allow_recording = True
        now = dt.now()
        username = getpass.getuser()
        file_name = f'{username}-{now.year}{now.month}{now.hour}{now.minute}{now.second}.mp3'
        self.file_name = f'{self.recording_path}/{file_name}.mp3'
        self.lb_status['text'] = 'Recording...'

        df = pd.DataFrame({'DsId': [self.ds_id], 'Orginaldatei': Path(self.file_name).name})
        self.write_thread.set_dataframe(df)
        self.record_thread.start()
        self.write_thread.start()

    def stop(self):
        if self.allow_recording:
            self.allow_recording = False
            self.lb_status['text'] = 'Ready'
            self.finished = True

            if self.record_thread.is_alive():
                self.record_thread.terminate()
                self.record_thread.join()

            if self.write_thread.is_alive():
                self.write_thread.terminate()
                self.write_thread.join()

            df = self.write_thread.result
            print(df)

    def close_window(self):
        if self.allow_recording:
            self.stop()

        SpeechRecordingGui.root.destroy()


gui = SpeechRecordingGui(recording_base_path='c:/temp/audio')
gui.root.mainloop()

# defines what happens when user closes window
gui.root.protocol('WM_DELETE_WINDOW', gui.close_window)
