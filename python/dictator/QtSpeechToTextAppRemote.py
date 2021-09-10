import sys
from PyQt6.QtWidgets import QApplication
from QtSpeechToTextApp import QtSpeechToTextApp
from SrcTranslationService import SrcTranslationService


def main():
    app = QApplication(sys.argv)
    translator = SrcTranslationService('127.0.0.1:8080')

    w = QtSpeechToTextApp(recording_base_path='c:/temp/audio', translator=translator)
    w.showMaximized()
    w.show()

    sys.exit(app.exec())


if __name__ == '__main__':
    main()
