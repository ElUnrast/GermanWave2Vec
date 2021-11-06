from PyQt6.QtWidgets import QTextEdit
from PyQt6.QtGui import QTextCursor


class OrigTextEditWidget(QTextEdit):
    def __init__(self, validation_app, translation_row: str):
        super().__init__()
        self.validation_app = validation_app
        self.translation_row = translation_row

    def focusOutEvent(self, event):
        idx = self.validation_app.curr_index

        orig_text_in_gui = self.validation_app.edit_rows[idx].toPlainText()
        ds_idx = self.validation_app.ds_index[idx]
        orig_text_in_ds = self.validation_app.ds['OriginalText'].values[ds_idx]

        if orig_text_in_gui != orig_text_in_ds:
            self.validation_app.ds['OriginalText'].values[ds_idx] = orig_text_in_gui

            if not self.validation_app.action[idx].startswith('exclude'):
                orig_translated_in_ds = self.validation_app.ds[self.translation_row].values[ds_idx]

                if orig_text_in_gui == orig_translated_in_ds:
                    self.validation_app.action[idx] = 'train9'
                    self.validation_app.ds['Action'].values[ds_idx] = 'train9'
                elif not self.validation_app.action[idx].startswith('ignore'):
                    self.validation_app.action[idx] = 'train8'
                    self.validation_app.ds['Action'].values[ds_idx] = 'train8'

        super().focusOutEvent(event)

    def focusInEvent(self, event):
        tmpCursor = self.textCursor()
        tmpCursor.movePosition(QTextCursor.MoveOperation.Right, QTextCursor.MoveMode.MoveAnchor, 4)
        self.setTextCursor(tmpCursor)

        self.validation_app.set_groupbox_color(self.validation_app.curr_index)
        super().focusInEvent(event)
        self.validation_app.curr_index = self.validation_app.get_current_row_index()
        self.validation_app.set_groupbox_color(self.validation_app.curr_index)
        self.validation_app.play()
