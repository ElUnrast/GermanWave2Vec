import sys
import pygame
from PyQt6.QtCore import Qt, QEvent
from PyQt6.QtWidgets import (
    QMainWindow,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QComboBox,
    QMessageBox,
    QScrollArea,
    QGroupBox,
    QTextEdit,
    QApplication
)
from PyQt6.QtGui import QColor, QFontMetrics
from SrcTextUtils import html_diff
from SnippetDatasets import SnippetDatasets
from OrigTextEditWidget import OrigTextEditWidget


class QtValidationApp(QMainWindow):
    def __init__(self, dataset_loader: SnippetDatasets):
        super().__init__()
        self.focus_color = QColor(int("A48111", 16))  # gold
        self.exclude_color = QColor(int("ffcccb", 16))  # Light red #ffcccb
        self.correct_color = QColor(int("90EE90", 16))  # lightgreen #90EE90
        self.rated_color = QColor(int("BB65E0", 16))  # lightgreen #90EE90
        self.corrected1_color = QColor(int("ADD8E6", 16))  # lightblue #ADD8E6
        self.corrected2_color = QColor(int("9999FF", 16))  #
        self.ignore_color = QColor(int("4ffdec ", 16))  # ignore
        self.my_datasets = dataset_loader

        pygame.init()
        pygame.mixer.init()
        self.playing = False
        self.halting = False

        all_ds_ids_set = set()
        all_ds_ids_set.update(list(self.my_datasets.local_datasets.keys()))
        all_ds_ids_set.update(list(self.my_datasets.extern_datasets.keys()))
        self.all_ds_ids = sorted(list(all_ds_ids_set))

        self.ds_id_combo = QComboBox()
        self.ds_id_combo.addItems(self.all_ds_ids)
        self.ds_id_combo.setCurrentIndex(0)
        self.ds_id_combo.currentIndexChanged.connect(self.on_dataset_change)

        top_widget_layout = QHBoxLayout()
        top_widget_layout.setSpacing(20)
        top_widget_layout.addWidget(self.ds_id_combo)

        self.scroll = QScrollArea()
        self.scroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOn)
        self.scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.scroll.setWidgetResizable(True)
        self.scroll.setWidget(self.create_sentence_list_widget())
        window_layout = QVBoxLayout()
        self.top_widget = QWidget()
        self.top_widget.setLayout(top_widget_layout)
        self.central_widget = QWidget()
        self.central_widget.setLayout(window_layout)
        window_layout.addWidget(self.top_widget)
        window_layout.addWidget(self.scroll)
        self.setCentralWidget(self.central_widget)
        self.curr_index = 0
        self.edit_rows[self.curr_index].setFocus()

    def create_sentence_list_widget(self):
        self.ds_id = self.ds_id_combo.currentText()
        wer = self.my_datasets.get_word_error_rate(self.ds_id)
        self.ds_epoche = wer['trained_epochs']
        self.wer = wer['wer']

        vbox = QVBoxLayout()
        self.ds = self.my_datasets.load_ds_content_translated_with_original(self.ds_id, prune=False)

        if not 'OriginalText' in self.ds.columns:
            self.ds['OriginalText'] = ' '
            self.ds['Action'] = 'validate'

        without_original = len(self.ds[self.ds.OriginalText.str.len() < 2])

        if not 'Sort1' in self.ds.columns:
            self.ds['Sort1'] = 0

        self.snipped_directory = self.my_datasets.get_snippet_directory(self.ds_id)
        all = len(self.ds)
        self.translation_row = 'Translated1' if 'Translated1' in self.ds.columns else 'Translated0'
        ds_problematic = self.ds[self.ds['OriginalText'] != self.ds[self.translation_row]]
        ds_problematic = ds_problematic.sort_values(['Action', 'Sort1'], ascending=[False, True])
        wrong = len(ds_problematic)
        self.setWindowTitle(f'Dataset Validation of {self.ds_id}, epoche {self.ds_epoche}, WER: {self.wer:3.4f}%, bad {wrong}, without original {without_original}')
        print(f'Use Snipped Directory: {self.snipped_directory} - {all}/{wrong} Wrong')
        self.action = []
        self.mp3_files = []
        self.rows = []
        self.ds_index = []
        self.edit_rows = []
        manuell_validated_train_actions = ['train7', 'train8', 'train9']
        count = 0

        for idx in range(len(ds_problematic)):
            aktion = ds_problematic.iloc[idx]['Action']
            length = ds_problematic.iloc[idx]['Length']

            if (wrong > 500) and (aktion.startswith('exclude') or ((aktion in manuell_validated_train_actions) and (count > 500))):
                continue

            if (wrong > 500) and (length > 120):
                continue

            ds_idx = ds_problematic.index[idx]
            self.ds_index.append(ds_idx)
            self.action.append(aktion)
            translated_text = ds_problematic.iloc[idx][self.translation_row]
            original_text = ds_problematic.iloc[idx]['OriginalText']
            file_name = ds_problematic.iloc[idx]['Datei']
            self.mp3_files.append(file_name)
            html, diff_value1 = html_diff(original_text, translated_text)
            self.ds['Sort1'].values[ds_idx] = diff_value1
            row = QGroupBox(title=f'{file_name} - {diff_value1}')

            if aktion.startswith('exclude') or (aktion in manuell_validated_train_actions):
                self.set_initial_groupbox_color(row, self.get_color_for_action(aktion))

            self.rows.append(row)
            row.setFocusPolicy(Qt.FocusPolicy.NoFocus)
            row.setAutoFillBackground(True)
            row_vbox = QVBoxLayout()
            row.setLayout(row_vbox)
            row.setProperty('index', count)
            diff = QTextEdit()
            diff.setFocusPolicy(Qt.FocusPolicy.NoFocus)
            diff.setHtml(html)
            diff.setReadOnly(True)
            diff.setProperty('index', count)
            font = diff.document().defaultFont()
            fontMetrics = QFontMetrics(font)
            textSize = fontMetrics.size(0, diff.toPlainText())
            h = textSize.height() + 10
            diff.setMinimumHeight(h)
            diff.setMaximumHeight(h)
            orig = OrigTextEditWidget(self, self.translation_row)
            orig.setPlainText(original_text)
            orig.setReadOnly(False)
            orig.setProperty('index', count)
            orig.setMinimumHeight(h)
            orig.setMaximumHeight(h)
            self.edit_rows.append(orig)
            row.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
            row_vbox.addWidget(diff)
            row_vbox.addWidget(orig)
            vbox.addWidget(row)
            count += 1

        result = QWidget()
        result.setLayout(vbox)
        return result

    def on_dataset_change(self):
        print(f'Dataset changed to: {self.ds_id_combo.currentText()}')
        # TODO: Speichern Abfrage und Neuaufbau der Liste
        self.scroll.setWidget(self.create_sentence_list_widget())

    def set_initial_groupbox_color(self, group_box: QGroupBox, color: QColor):
        p = group_box.palette()
        p.setColor(group_box.backgroundRole(), color)
        group_box.setPalette(p)

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
        elif action == 'ignore':
            return self.ignore_color
        elif action == 'rated':
            return self.rated_color

        return self.correct_color

    def get_color_for_action(self, action) -> QColor:
        if action.startswith('exclude'):
            return self.exclude_color
        elif action == 'ignore':
            return self.ignore_color
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
            if e.key() == Qt.Key.Key_Space:
                if self.pause:
                    self.unpause()
                else:
                    self.pause()
            if e.key() == Qt.Key.Key_P:
                self.play()
            elif e.key() == Qt.Key.Key_S:
                self.save()
            elif e.key() == Qt.Key.Key_T:
                self.copy_translated_to_original()
            elif e.key() == Qt.Key.Key_Return:
                # Copy and accept translation
                self.copy_translated_to_original()
                self.play_next()
            elif e.key() == Qt.Key.Key_PageUp:
                self.play_previos()
            elif e.key() == Qt.Key.Key_PageDown:
                idx = self.get_current_row_index()
                ds_idx = self.ds_index[idx]

                if not self.action[idx].startswith('exclude'):
                    if (not self.action[idx] == 'train8') and (not self.action[idx] == 'train9'):
                        self.action[idx] = 'train7'
                        self.ds['Action'].values[ds_idx] = 'train7'

                self.play_next()
            elif e.key() == Qt.Key.Key_E:
                # Toggle Exclude
                idx = self.get_current_row_index()
                ds_idx = self.ds_index[idx]

                if self.action[idx] == 'exclude9':
                    self.action[idx] = 'train'
                    self.ds['Action'].values[ds_idx] = 'train'
                else:
                    self.action[idx] = 'exclude9'
                    self.ds['Action'].values[ds_idx] = 'exclude9'

                self.play_next()
            elif e.key() == Qt.Key.Key_I:
                idx = self.get_current_row_index()
                ds_idx = self.ds_index[idx]

                if self.action[idx] == 'ignore':
                    self.ds['Action'].values[ds_idx] = 'train'
                else:
                    self.action[idx] = 'ignore'
                    self.ds['Action'].values[ds_idx] = 'ignore'

                self.play_next()
            elif e.key() == 92:
                # Toggle Ratet
                idx = self.get_current_row_index()
                ds_idx = self.ds_index[idx]

                if self.action[idx] == 'rated':
                    self.action[idx] = 'train'
                    self.ds['Action'].values[ds_idx] = 'train'
                else:
                    self.action[idx] = 'rated'
                    self.ds['Action'].values[ds_idx] = 'rated'

                    if not self.edit_rows[idx].toPlainText().strip():
                        translated_text = self.ds[self.translation_row].values[ds_idx]
                        self.edit_rows[idx].setPlainText(translated_text)
                        self.play_next()

    def copy_translated_to_original(self):
        idx = self.get_current_row_index()
        ds_idx = self.ds_index[idx]
        translated_text = self.ds[self.translation_row].values[ds_idx]

        if isinstance(translated_text, str):
            # print(f'setting original text (idx = {idx}, ds_idx = {ds_idx}): {translated_text}')
            # print(f'mp3: {self.mp3_files[idx]}, orig_mp3: {self.ds["Datei"].values[ds_idx]})')
            self.edit_rows[idx].setPlainText(translated_text)

    def save(self):
        print('saving dataset')
        self.my_datasets.save_content_translated_with_original(self.ds_id, self.ds, self.ds_epoche)

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

    def play(self):
        if pygame.mixer.music.get_busy():
            pygame.mixer.music.stop()

        curr_index = self.get_current_row_index()
        print(f'Play index {curr_index}')
        pygame.mixer.music.load(f'{self.snipped_directory}/{self.mp3_files[curr_index]}')
        # var.set(play_list.get(tkr.ACTIVE))
        pygame.mixer.music.play()
        self.playing = True
        self.halting = False

    def stop(self):
        pygame.mixer.music.stop()
        self.playing = False
        self.halting = False

    def pause(self):
        if not self.playing:
            pygame.mixer.music.pause()
            self.playing = False
            self.halting = True

    def unpause(self):
        if self.halting:
            pygame.mixer.music.unpause()
            self.halting = False
            self.playing = True
        else:
            self.play()

    def closeEvent(self, event):
        close_msg_dialog = QMessageBox(parent=self)
        close_msg_dialog.setIcon(QMessageBox.Icon.Question)
        close_msg_dialog.setWindowTitle("Anwendung beenden.")
        close_msg_dialog.setText("Sollen die Änderungen gespeichert werden?")
        close_msg_dialog.setStandardButtons(
            QMessageBox.StandardButton.Save
            | QMessageBox.StandardButton.Discard
            | QMessageBox.StandardButton.Abort
        )
        close_msg_dialog.setDefaultButton(QMessageBox.StandardButton.Save)
        close_msg_dialog.exec()

        if close_msg_dialog == QMessageBox.StandardButton.Abort:
            event.ignore()
        else:
            if close_msg_dialog == QMessageBox.StandardButton.Save:
                self.save()

            event.accept()


def main():
    dataset_loader = SnippetDatasets(False, '//matlab3/D/NLP-Data/audio', 'C:/gitviews/GermanWave2Vec')
    # dataset_loader = SnippetDatasets(False, local_audio_base_dir='C:/temp/audio')
    print(f'Local Datasets: {dataset_loader.local_datasets}')
    app = QApplication(sys.argv)
    w = QtValidationApp(dataset_loader)
    w.showMaximized()
    w.show()

    sys.exit(app.exec())


if __name__ == '__main__':
    main()
