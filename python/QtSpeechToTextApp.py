import sys
from PyQt6.QtCore import *
from PyQt6.QtWidgets import *
from PyQt6.QtGui import *
import getpass
import os
import re
import sounddevice as sd
import pandas as pd
from SpeechComanndProcessor import SpeechEventHandler, SpeechComanndSatzzeichenProcessor
from SpeechComanndProcessor import SpeechComanndFormatProcessor, SpeechComanndBuchstabierenProcessor
from SpeechComanndProcessor import SpeechComanndZahlProcessor
from queue import Queue
from datetime import datetime as dt
from pathlib import Path
from GermanSpeechToTextTranslaterBase import GermanSpeechToTextTranslaterBase
from RecordingThread import RecordingThread
from SplitterThread import SplitterThread
from time import sleep
from threading import Thread
from threadutil import run_in_main_thread


class SpeechTextEvent(QObject):
    progressSignal = pyqtSignal(str)


class SpeechFromatEvent(QObject):
    progressSignal = pyqtSignal(bool)


class QtSpeechToTextApp(QMainWindow, SpeechEventHandler):
    def __init__(self, recording_base_path: str, translator: GermanSpeechToTextTranslaterBase = None):
        super().__init__()
        self.translator = translator
        self.allow_recording = False
        self.last_processor = None
        # speakers = sc.all_speakers()
        # self.speaker = sc.default_speaker()
        # mics = sc.all_microphones()
        # self.mic = sc.default_microphone()
        self.mics = sd.query_devices(kind='input')
        self.frame_rate = 16_000
        self.frame_duration_ms = 30
        self.channels = 1
        self.frame_size = int((self.frame_rate / 1000) * self.frame_duration_ms * 2)
        self._init_directories(recording_base_path=recording_base_path)

        self.audio_in_queue = Queue()
        self.translated_text_queue = Queue()
        self.finished = False
        self.file_name = None  # wird in start gesetzt
        self.record_thread = None
        self.write_thread = None

        self.speech_command_processors = [
            SpeechComanndSatzzeichenProcessor(speech_event_handler=self),
            SpeechComanndFormatProcessor(speech_event_handler=self),
            SpeechComanndBuchstabierenProcessor(speech_event_handler=self),
            SpeechComanndZahlProcessor(speech_event_handler=self),
        ]

        self.init_gui()
        self.display_thread = Thread(target=self.display_new_messages, daemon=True)
        self.display_thread.start()

    def init_gui(self):
        self.record_on_icon = QIcon(QPixmap("microphone_red.png"))
        # self.record_on_icon.addPixmap(QPixmap("microphone_red_small.png"), QIcon.Mode.Normal, QIcon.State.On)
        self.record_off_icon = QIcon(QPixmap("microphone_blue.png"))
        # self.record_off_icon.addPixmap(QPixmap("microphone_blue_small.png"), QIcon.Mode.Normal, QIcon.State.Off)

        self.statusBar()
        self.central_widget = QWidget()
        vbox_layout = QVBoxLayout()
        self.central_widget.setLayout(vbox_layout)

        hbox_layout = QHBoxLayout()
        hbox_layout.setSpacing(20)
        self.record_button = QPushButton(parent=self)  # QToolButton()
        self.record_button.setText("Record")
        self.record_button.setCheckable(True)
        self.record_button.setChecked(False)
        self.record_button.setIcon(self.record_off_icon)
        self.record_button.setIconSize(QSize(32, 32))
        self.record_button.setGeometry(QRect(34, 34, 34, 34))
        self.record_button.clicked.connect(self.record)
        hbox_layout.addWidget(self.record_button)
        # Bei ToolButton könnte man noch sowas machen
        # self.record_button.setArrowType(Qt.RightArrow)
        # self.record_button.setAutoRaise(True)
        # self.record_button.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonIconOnly)

        hbox_layout.addWidget(QLabel('Benutzer'))
        self.user_combo = QComboBox()
        self.user_combo.addItem(getpass.getuser())
        self.user_combo.currentIndexChanged.connect(self.on_user_change)
        hbox_layout.addWidget(self.user_combo)

        self.satzzeichen_check = QCheckBox(parent=self)
        self.satzzeichen_check.setText('Satzzeichen diktieren')
        hbox_layout.addWidget(self.satzzeichen_check)

        self.formated_text_area = QTextBrowser()
        self.formated_text_area.setAcceptRichText(True)
        self.formated_text_area.setFocusPolicy(Qt.FocusPolicy.ClickFocus)
        self.received_text_area = QPlainTextEdit()
        self.received_text_area.setFocusPolicy(Qt.FocusPolicy.NoFocus)

        self.top_widget = QWidget()
        self.top_widget.setLayout(hbox_layout)
        vbox_layout.addWidget(self.top_widget)
        vbox_layout.addWidget(self.formated_text_area)
        vbox_layout.addWidget(self.received_text_area)
        self.setCentralWidget(self.central_widget)
        self.translated_text_queue = Queue()
        self._append_received_text = run_in_main_thread(self.received_text_area.appendPlainText)

        self.formated_text_emitter = SpeechTextEvent()
        self.formated_text_emitter.progressSignal.connect(self._insert_formatted_text)
        self.text_bold_format_event_emmiter = SpeechFromatEvent()
        self.text_bold_format_event_emmiter.progressSignal.connect(self._set_bold)
        self.text_italic_format_event_emmiter = SpeechFromatEvent()
        self.text_italic_format_event_emmiter.progressSignal.connect(self._set_italic)

    def display_new_messages(self):
        while True:
            received_text = self.translated_text_queue.get()
            print(f'Got translation from queue: {received_text}')

            if not received_text:
                print('Exiting Display Thread')
                break

            self._append_received_text(received_text)
            self.handle_speech_event(received_text)

    def is_formated_text_at_block_start(self):
        return self.formated_text_area.textCursor().atBlockStart()

    def get_last_proccessor(self):
        return self.last_processor

    def handle_speech_event(self, text: str):
        txt = text.strip()

        if txt:
            for processor in self.speech_command_processors:
                if processor.process(text):
                    self.last_processor = processor
                    return

            self.last_processor = None
            formatted_text = text if self.is_formated_text_at_block_start() else f' {text}'
            self.append_formatted_text(formatted_text)

    def append_formatted_text(self, text):
        if text:
            print(f'insert text: {text}')
            self.formated_text_emitter.progressSignal.emit(text)

    def _insert_formatted_text(self, text):
        self.formated_text_area.insertPlainText(text)

    def set_bold(self, b: bool):
        self.text_bold_format_event_emmiter.progressSignal.emit(b)

    def _set_bold(self, b: bool):
        self.formated_text_area.setFontWeight(QFont.Weight.Bold if b else QFont.Weight.Normal)

    def set_italic(self, b: bool):
        self.text_italic_format_event_emmiter.progressSignal.emit(b)

    def _set_italic(self, b: bool):
        self.formated_text_area.setFontItalic(b)

    def on_change_base_directory(self):
        home_dir = str(Path.home())
        self._init_directories(QFileDialog.getExistingDirectory(parent=self, caption='Record Base Directory', directory=home_dir))

    def on_user_change(self):
        print(f'User changed to: {self.user_combo.currentText()}')

    def _init_directories(self, recording_base_path):
        self.ds_id = 'Recording'
        self.recording_base_path = recording_base_path
        self.recording_path = f'{self.recording_base_path}/Full{self.ds_id}'
        self.snippet_path = f'{self.recording_base_path}/{self.ds_id}'
        self.ds_filename = f'{self.snippet_path}/content-translated-with_original.csv'

        if os.path.isfile(self.ds_filename):
            self.pandas_df = pd.read_csv(self.ds_filename, sep=';')
        else:
            self.pandas_df = pd.DataFrame()

    def record(self):
        if self.record_button.isChecked():
            self.record_button.setIcon(self.record_on_icon)
            self.statusBar().showMessage("Recording....")
            self.start()
        else:
            self.statusBar().showMessage("Stop recording...")
            self.stop()
            self.statusBar().showMessage("Not recording")
            self.record_button.setIcon(self.record_off_icon)

    def start(self):
        if not self.allow_recording:
            self.allow_recording = True
            now = dt.now()
            username = self.user_combo.currentText()
            file_name = f'{username}-{now.year}{now.month:0>2d}{now.hour:0>2d}{now.minute:0>2d}{now.second:0>2d}.mp3'
            self.file_name = f'{self.recording_path}/{file_name}'

            df = pd.DataFrame({'DsId': [self.ds_id], 'Orginaldatei': Path(self.file_name).name})
            self.write_thread = SplitterThread(
                snippet_path=self.snippet_path,
                audio_in_queue=self.audio_in_queue,
                vad_level=3,
                translator=self.translator,
                translated_text_queue=self.translated_text_queue
            )
            self.write_thread.set_dataframe(df)
            self.write_thread.start()
            self.record_thread = RecordingThread(self.audio_in_queue)
            self.record_thread.start()

    def stop(self):
        if self.allow_recording:
            self.allow_recording = False
            self.finished = True

            if self.record_thread.is_alive():
                self.record_thread.terminate()
                self.record_thread.join()

            if self.write_thread.is_alive():
                self.write_thread.join()

            snippet_df = self.write_thread.result
            print(snippet_df)
            snippet_df['OriginalText'] = ' '
            snippet_df['Action'] = 'validate'
            self.pandas_df = self.pandas_df.append(snippet_df, ignore_index=True)
            print(f'Saving: {self.ds_filename}')
            self.pandas_df.to_csv(self.ds_filename, sep=';', index=False)

    def closeEvent(self, event):
        close = QMessageBox.question(
            self,
            "QUIT",
            "Are you sure want to stop process?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )

        if close == QMessageBox.StandardButton.Yes:
            self.close_window()
            event.accept()
        else:
            event.ignore()

    def close_window(self):
        if self.allow_recording:
            self.stop()
            self.translated_text_queue.put(None)
            self.display_thread.join()


def main():
    app = QApplication(sys.argv)
    model_name = 'c:/share/NLP-Models/GermanWave2Vec/trained_model'
    translator = GermanSpeechToTextTranslaterBase(model_name=model_name, device='cpu')
    w = QtSpeechToTextApp(recording_base_path='c:/temp/audio', translator=translator)
    w.showMaximized()
    w.show()

    sys.exit(app.exec())


if __name__ == '__main__':
    main()
