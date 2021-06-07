import difflib
from IPython.display import Audio, display
from ipywidgets import widgets
from .SnippetDatasets import calc_wer
from tqdm.notebook import tqdm_notebook

class InvisibleAudio(Audio):
    def _repr_html_(self):
        audio = super()._repr_html_()
        audio = audio.replace('<audio', f'<audio onended="this.parentNode.removeChild(this)"')
        return f'<div style="display:none">{audio}</div>'


class GermanSpeechDatasetWidgetFactory:
    def __init__(self, dataset_loader, ds_to_use=[]):
        self.my_datasets = dataset_loader

        all_ds_ids = set()
        all_ds_ids.update(list(self.my_datasets.local_datasets.keys()))
        all_ds_ids.update(list(self.my_datasets.extern_datasets.keys()))

        self.ds_checkbox_items = []

        for ds_id in sorted(all_ds_ids):
            self.ds_checkbox_items.append(widgets.Checkbox(
                value=ds_id in ds_to_use, 
                description=ds_id, 
                disabled=False, 
                indent=False
            ))

    def get_used_datasets(self):
        result = []

        for item in self.ds_checkbox_items:
            if item.value:
                result.append(item.description)

        return result

    def create_dataset_choice_widget(self):
        return widgets.Box(self.ds_checkbox_items)

    def create_validation_tab_widget(self):
        print('Validate')
        tab = widgets.Tab()
        tab_titles = []
        tab_children = []

        for ds_id in self.get_used_datasets():
            ds = self.my_datasets.load_ds_content_translated_with_original(ds_id)
            wer = 100 * calc_wer(ds)
            tab_titles.append(f'{ds_id} - {wer:3.4f}')
            tab_children.append(self._create_diff_content(ds_id, ds))

        tab.children = tab_children

        for idx, name in enumerate(tab_titles):
            tab.set_title(idx, name)

        return tab

    def play_audio_file(self, audio_file_name=None, audio_url=None):
        if audio_file_name:
            display(InvisibleAudio(filename=audio_file_name, autoplay=True))
        elif audio_url:
            display(InvisibleAudio(url=audio_url, autoplay=True))

    def _html_diff(self, text, n_text):
        """
        http://stackoverflow.com/a/788780
        Unify operations between two compared strings seqm is a difflib.
        SequenceMatcher instance whose a & b are strings
        """
        seqm = difflib.SequenceMatcher(None, text, n_text)
        output= []
        output.append('<p style="font-size:60%;">')
        for opcode, a0, a1, b0, b1 in seqm.get_opcodes():
            if opcode == 'equal':
                output.append(seqm.a[a0:a1])
            elif opcode == 'insert':
                output.append("<font color=red>^" + seqm.b[b0:b1] + "</font>")
            elif opcode == 'delete':
                output.append("<font color=blue>^" + seqm.a[a0:a1] + "</font>")
            elif opcode == 'replace':
                # seqm.a[a0:a1] -> seqm.b[b0:b1]
                output.append("<font color=green>^" + seqm.b[b0:b1] + "</font>")
            else:
                raise RuntimeError("unexpected opcode")

        output.append('</p>')
        return ''.join(output)

    def _create_diff_row(self, audio_file, translated_text, original_text ):
        diff_widget = widgets.HTML(value=self._html_diff(original_text, translated_text))
        button_widget = widgets.Button(description='Play')

        def on_play_button_clicked(b):
            self.play_audio_file(audio_file_name=audio_file)

        button_widget.on_click(on_play_button_clicked)
        return widgets.HBox([button_widget, diff_widget])

    def _create_diff_content(self, ds_id, ds):
        snipped_directory = self.my_datasets.get_snippet_directory(ds_id)
        all = len(ds)
        translation_row = 'Translated1' if 'Translated1' in ds.columns else 'Translated0'
        ds = ds[ds['OriginalText'] != ds[translation_row]]
        wrong = len(ds)
        print(f'Use Snipped Directory: {snipped_directory} - {all}/{wrong} Wrong')
        rows = []
        max_len = min(50, len(ds))

        for idx in tqdm_notebook(range(max_len)):
            rows.append(self._create_diff_row(
                f'{snipped_directory}/{ds.iloc[idx]["Datei"]}', 
                ds.iloc[idx]['OriginalText'], 
                ds.iloc[idx][translation_row]
            ))

        return widgets.VBox(rows)

