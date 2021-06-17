import sys
import pygame
from PyQt6.QtCore import *
from PyQt6.QtWidgets import *
from PyQt6.QtGui import *
from SrcTextUtils import html_diff
from SnippetDatasets import SnippetDatasets


class MyApp(QMainWindow):
    def __init__(self, dataset_loader: SnippetDatasets):
        super().__init__()
        self.focus_color = QColor(int("A48111", 16))  # gold
        self.exclude_color = QColor(int("ffcccb", 16))  # Light red #ffcccb
        self.correct_color = QColor(int("90EE90", 16))  # lightgreen #90EE90
        self.corrected1_color = QColor(int("ADD8E6", 16))  # lightblue #ADD8E6
        self.corrected2_color = QColor(int("9999FF", 16))  #
        self.my_datasets = dataset_loader
        self.setWindowTitle('Dataset Validation')
        self.window_width, self.window_height = 700, 100
        self.resize(self.window_width, self.window_height)

        pygame.init()
        pygame.mixer.init()

        class OrigTextEdit(QTextEdit):
            def __init__(self, app: MyApp):
                super(OrigTextEdit, self).__init__()
                self.myApp = app

            def focusOutEvent(self, event):
                idx = self.myApp.curr_index

                if not self.myApp.action[idx].endswith('9'):
                    orig_text_in_gui = self.myApp.edit_rows[idx].toPlainText()
                    ds_idx = self.myApp.ds_index[idx]
                    orig_text_in_ds = self.myApp.ds['OriginalText'].values[ds_idx]

                    if orig_text_in_gui != orig_text_in_ds:
                        self.myApp.action[idx] = 'train8'
                        self.myApp.ds['Action'].values[ds_idx] = 'train8'
                        self.myApp.ds['OriginalText'].values[ds_idx] = orig_text_in_gui

                super().focusOutEvent(event)

            def focusInEvent(self, event):
                if event.type() == QEvent.Type.FocusIn:
                    print('Focus in erkannt')
                elif event.type() == QEvent.Type.FocusOut:
                    print('Focus out erkannt')

                ## old_textedit = self.myApp.get_current_text_edit()
                tmpCursor = self.textCursor()
                tmpCursor.movePosition(QTextCursor.MoveOperation.Right, QTextCursor.MoveMode.MoveAnchor, 4)
                self.setTextCursor(tmpCursor)

                print(f'old current index: {self.myApp.curr_index}')
                self.myApp.set_groupbox_color(self.myApp.curr_index)
                # super(OrigTextEdit, self).focusInEvent(event)
                super().focusInEvent(event)
                self.myApp.curr_index = self.myApp.get_current_row_index()
                print(f'new current index: {self.myApp.curr_index}')
                self.myApp.set_groupbox_color(self.myApp.curr_index)
                self.myApp.play()

        self.scroll = QScrollArea()
        self.widget = QWidget()
        self.vbox = QVBoxLayout()
        self.widget.setLayout(self.vbox)

        self.ds_id = 'HP4-FvM'
        wer = self.my_datasets.get_word_error_rate(self.ds_id)
        self.ds_epoche = wer['trained_epochs']
        self.ds = self.my_datasets.load_ds_content_translated_with_original(self.ds_id)
        self.snipped_directory = self.my_datasets.get_snippet_directory(self.ds_id)
        all = len(self.ds)
        self.translation_row = 'Translated1' if 'Translated1' in self.ds.columns else 'Translated0'
        ds_problematic = self.ds[self.ds['OriginalText'] != self.ds[self.translation_row]]
        wrong = len(ds_problematic)
        print(f'Use Snipped Directory: {self.snipped_directory} - {all}/{wrong} Wrong')
        self.action = []
        self.curr_index = 0
        self.mp3_files = []
        self.rows = []
        self.ds_index = []
        self.edit_rows = []

        for idx in range(len(ds_problematic)):
            self.ds_index.append(ds_problematic.index[idx])
            self.action.append(ds_problematic.iloc[idx]['Action'])
            translated_text = ds_problematic.iloc[idx][self.translation_row]
            original_text = ds_problematic.iloc[idx]['OriginalText']
            file_name = ds_problematic.iloc[idx]['Datei']
            self.mp3_files.append(file_name)
            html = html_diff(original_text, translated_text)
            row = QGroupBox(title=file_name)
            self.rows.append(row)
            row.setFocusPolicy(Qt.FocusPolicy.NoFocus)
            row.setAutoFillBackground(True)
            row_vbox = QVBoxLayout()
            row.setLayout(row_vbox)
            row.setProperty('index', idx)
            diff = QTextEdit()
            diff.setFocusPolicy(Qt.FocusPolicy.NoFocus)
            diff.setHtml(html)
            diff.setReadOnly(True)
            diff.setProperty('index', idx)
            font = diff.document().defaultFont()
            fontMetrics = QFontMetrics(font)
            textSize = fontMetrics.size(0, diff.toPlainText())
            h = textSize.height() + 10
            diff.setMinimumHeight(h)
            diff.setMaximumHeight(h)
            orig = OrigTextEdit(self)
            orig.setPlainText(original_text)
            orig.setReadOnly(False)
            orig.setProperty('index', idx)
            orig.setMinimumHeight(h)
            orig.setMaximumHeight(h)
            self.edit_rows.append(orig)
            row.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
            row_vbox.addWidget(diff)
            row_vbox.addWidget(orig)
            self.vbox.addWidget(row)

        # Scroll Area Properties
        self.scroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOn)
        self.scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.scroll.setWidgetResizable(True)
        self.scroll.setWidget(self.widget)
        self.setCentralWidget(self.scroll)
        self.edit_rows[self.curr_index].setFocus()

    def set_groupbox_color(self, idx):
        group_box = self.rows[idx]
        p = group_box.palette()
        p.setColor(group_box.backgroundRole(), self.get_color_for_index(idx))
        group_box.setPalette(p)

    def get_color_for_index(self, idx):
        if idx == self.get_current_row_index():
            return self.focus_color

        action = self.action[idx]

        if action.startswith('exclude'):
            return self.exclude_color
        elif action == 'train9':
            return self.corrected1_color
        elif action == 'train8':
            return self.corrected2_color

        return self.correct_color

    def keyPressEvent(self, e: QEvent):
        if e.key() == Qt.Key.Key_Escape.value:
            # TODO save
            self.close()

        if e.modifiers() & Qt.KeyboardModifier.ControlModifier:
            if e.key() == Qt.Key.Key_P:
                self.play()
            elif e.key() == Qt.Key.Key_S:
                print('saving dataset')
                self.my_datasets.save_content_translated_with_original(self.ds_id, self.ds, self.ds_epoche)
            elif e.key() == Qt.Key.Key_T:
                idx = self.get_current_row_index()
                ds_idx = self.ds_index[idx]
                self.action[idx] = 'train9'
                self.ds['Action'].values[ds_idx] = 'train9'
                translated_text = self.ds[self.translation_row].values[ds_idx]
                self.edit_rows[idx].setPlainText(translated_text)
                self.ds['OriginalText'].values[ds_idx] = translated_text
                self.play_next()
            elif e.key() == Qt.Key.Key_PageUp:
                self.play_previos()
            elif e.key() == Qt.Key.Key_PageDown:
                self.play_next()
            elif e.key() == Qt.Key.Key_E:
                idx = self.get_current_row_index()
                self.action[idx] = 'exclude9'
                self.ds['Action'].values[self.ds_index[idx]] = 'exclude9'
                self.play_next()

    def play_previos(self):
        previous_row = self.edit_rows[self.get_previous_row_index()]
        self.scroll.ensureWidgetVisible(previous_row, xMargin=100)
        previous_row.setFocus()

    def play_next(self):
        next_row = self.edit_rows[self.get_next_row_index()]
        self.scroll.ensureWidgetVisible(next_row, xMargin=100)
        next_row.setFocus()

    def get_current_row_index(self):
        return self.focusWidget().property('index')

    def get_next_row_index(self):
        return (self.get_current_row_index() + 1) % len(self.edit_rows)

    def get_previous_row_index(self):
        return (self.get_current_row_index() + len(self.edit_rows) - 1) % len(self.edit_rows)

    def get_current_text_edit(self):
        return self.edit_rows[self.get_next_row_index()]

    def play(self):
        if pygame.mixer.music.get_busy():
            pygame.mixer.music.stop()

        curr_index = self.get_current_row_index()
        print(f'Play index {curr_index}')
        pygame.mixer.music.load(f'{self.snipped_directory}/{self.mp3_files[curr_index]}')
        # var.set(play_list.get(tkr.ACTIVE))
        pygame.mixer.music.play()

    def stop(self):
        pygame.mixer.music.stop()

    def pause(self):
        pygame.mixer.music.pause()

    def unpause(self):
        pygame.mixer.music.unpause()


def main():
    app = QApplication(sys.argv)
    dataset_loader = SnippetDatasets(False, '//matlab3/D/NLP-Data/audio', 'C:/gitviews/GermanWave2Vec')
    w = MyApp(dataset_loader)
    w.showMaximized()
    w.show()

    sys.exit(app.exec())


if __name__ == '__main__':
    main()
