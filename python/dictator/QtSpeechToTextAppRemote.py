import os
import sys
from pathlib import Path
from PyQt6.QtWidgets import QApplication
from QtSpeechToTextApp import QtSpeechToTextApp
from SrcTranslationService import SrcTranslationService


def main():
    if not os.environ['QT_QPA_PLATFORM_PLUGIN_PATH']:
        os.environ['QT_QPA_PLATFORM_PLUGIN_PATH'] = '.'

    app = QApplication(sys.argv)
    # translator = SrcTranslationService('127.0.0.1:8080')
    translator = SrcTranslationService('192.168.9.105:8080')

    Path("c:/temp/audio").mkdir(parents=True, exist_ok=True)
    w = QtSpeechToTextApp(recording_base_path='c:/temp/audio', translator=translator)
    w.showMaximized()
    w.show()

    sys.exit(app.exec())


if __name__ == '__main__':
    main()
