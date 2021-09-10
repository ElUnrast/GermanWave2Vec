import difflib
from IPython.display import Audio, display
from ipywidgets import widgets, Textarea, Layout, HBox, VBox, Button
from dictator.SnippetDatasets import calc_wer
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

        self.manipulated_text = {}

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
            # get global version of unmanipulated df for reference
            self.manipulated_text[ds_id] = ds.copy()
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
        output = []
        output.append('<p style="font-size:100%;">')
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

    def _create_diff_row(self, ds_id, audio_file, translated_text, original_text, id):
        html_diff_text, _ = self._html_diff(original_text, translated_text)
        diff_widget = widgets.HTML(value=html_diff_text)
        input_widget = Textarea(value=original_text, placeholder='Original Text', layout=Layout(width='60vw'))
        play_button_widget = widgets.Button(description='Play')
        delete_button_widget = widgets.Button(description='Delete')
        translate_to_original_button = Button(description='To Original')

        df = self.manipulated_text[ds_id]

        def on_play_button_clicked(c):
            self.play_audio_file(audio_file_name=audio_file)

        def input_widget_state(c):
            """
            Updates the original Text of the DF according to changes made in the input field

                paramteters:
                            c['new']: new text
                            c['name']: type of state change (either 'value' or '_property_lock')
            """
            if c['name'] == 'value':
                index_of_diff_row = list(df['Datei'].values).index(id)
                changed_text = c['new']

                # replace original substring with edited one
                df['OriginalText'].values[index_of_diff_row] = changed_text
                df['Action'].values[index_of_diff_row] = 'train10'

        def on_delete_button_clicked(c):
            index_of_diff_row = list(df['Datei'].values).index(id)
            df['Action'].values[index_of_diff_row] = 'exclude10'
            input_widget.layout.border = '2px solid #FF0000'
            print(df['Action'].values[index_of_diff_row])

        def translate_to_original(c):
            index_of_diff_row = list(df['Datei'].values).index(id)
            translation_column = 'Translated1' if 'Translated1' in df.columns else 'Translated0'
            translated_text = df[translation_column].values[index_of_diff_row]
            df['OriginalText'].values[index_of_diff_row] = translated_text
            input_widget.value = translated_text
            print(df['OriginalText'].values[index_of_diff_row])

        def create_layout():
            box_layout = Layout(
                width='100%',
                display='flex',
                flex_flow='row',
                justify_content='flex-start',
                align_items='center',
                border='solid',
                min_height='100px',
                max_height='auto',
                margin='5px 10px 5px 10px',
                padding='10px 0 10px 0'
            )

            button_box = VBox(
                [play_button_widget, delete_button_widget, translate_to_original_button],
                layout=Layout(width='20vw')
            )
            text_box = VBox(
                [diff_widget, input_widget],
                layout=Layout(width='70vw')
            )

            return HBox(children=[button_box, text_box], layout=box_layout)

        input_widget.observe(input_widget_state)
        play_button_widget.on_click(on_play_button_clicked)
        delete_button_widget.on_click(on_delete_button_clicked)
        translate_to_original_button.on_click(translate_to_original)

        return create_layout()

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
                ds_id,
                f'{snipped_directory}/{ds.iloc[idx]["Datei"]}',
                ds.iloc[idx][translation_row],
                ds.iloc[idx]['OriginalText'],
                ds.iloc[idx]['Datei']
            ))

        return widgets.VBox(rows)
