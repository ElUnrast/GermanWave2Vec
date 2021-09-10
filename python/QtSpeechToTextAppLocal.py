import sys
from PyQt6.QtWidgets import QApplication
from dictator.QtSpeechToTextApp import QtSpeechToTextApp
from GermanSpeechToTextTranslaterBase import GermanSpeechToTextTranslaterBase


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
