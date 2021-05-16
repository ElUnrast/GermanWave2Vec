import os
import glob
import pandas as pd
import jiwer
import json
# from jiwer import wer
from zipfile import ZipFile


# Known Snippet Directories with content.csv, content-with_original.csv
class SnippetDatasets:
    def __init__(self, run_on_colab, local_audio_base_dir, git_repository=None, extern_audio_base_dir=None):
        self.local_datasets = {}
        self.extern_datasets = {}
        self.used_datasets = []
        self.local_audio_base_dir = local_audio_base_dir
        self.extern_audio_base_dir = extern_audio_base_dir
        self.git_repository = git_repository

        for dir in self.directories_with_content(local_audio_base_dir):
            self.local_datasets[os.path.basename(dir)] = dir

        if extern_audio_base_dir:
            for zip_file in glob.glob(f'{extern_audio_base_dir}/*.zip'):
                ds_id = os.path.basename(zip_file)[0:-4]
                self.extern_datasets[ds_id] = zip_file

    def directories_with_content(self, dir):
        result = []

        if self.has_content(dir) or self.has_content_original(dir):
            result.append(dir)
        else:
            sub_dir = [f'{dir}/{d}' for d in os.listdir(dir) if os.path.isdir(f'{dir}/{d}')]

            for d in sub_dir:
                result.extend(self.directories_with_content(d))

        return result

    def use_datasets(self, datasets_to_use):
        if isinstance(datasets_to_use, list):
            for ds in datasets_to_use:
                self.use_dataset(ds)
        else:
            self.use_dataset(datasets_to_use)

    def use_dataset(self, ds_id_to_use):
        if ds_id_to_use in self.local_datasets:
            self.used_datasets.append(ds_id_to_use)
        elif ds_id_to_use in self.extern_datasets:
            # TODO: Falls nicht genug Speicher da ist sollten nicht verwendete externe Datasets,
            #       welche lokal vorhanden sind, lokal gelöscht werden.
            zip_file_name = f'{self.extern_audio_base_dir}/{ds_id_to_use}.zip'
            print(f'Download and extract {zip_file_name} from gdrive')

            with ZipFile(zip_file_name, 'r') as zip:
                zip.extractall(self.local_audio_base_dir)

            self.local_datasets[ds_id_to_use] = f'{self.local_audio_base_dir}/{ds_id_to_use}'
            self.used_datasets.append(ds_id_to_use)

    def has_content(self, id_or_directory):
        return os.path.isfile(f'{self._get_directory(id_or_directory)}/content.csv')

    def has_content_original(self, id_or_directory):
        return os.path.isfile(f'{self._get_directory(id_or_directory)}/content-with_original.csv')

    def has_translation(self, id_or_directory):
        return os.path.isfile(f'{self._get_directory(id_or_directory)}/content-translated.csv')

    def has_translation_with_original(self, id_or_directory):
        return os.path.isfile(f'{self._get_directory(id_or_directory)}/content-translated-with_original.csv')

    def needs_translation(self, directory):
        if self.has_content(directory):
            return not self.has_translation(directory)
        else:
            return not self.has_translation_with_original(directory)

    def get_snippet_directory(self, ds_id):
        return self.local_datasets[ds_id]

    def get_ds_git_directory(self, ds_id):
        if self.git_repository:
            snippet_directory = self.get_snippet_directory(ds_id)

            if snippet_directory:
                return f'{self.git_repository}/datasets/{ds_id}'

        return None;

    def save_content_translated_with_original(self, ds_id, pandas_df):
        git_directory = self.get_ds_git_directory(ds_id)
        
        if git_directory:
            pandas_df.to_csv(f'{git_directory}/content-translated-with_original.csv', sep=';', index=False)
        else:
            mp3_dir = self.get_snippet_directory(ds_id)
            pandas_df.to_csv(f'{mp3_dir}/content-translated-with_original.csv', sep=';', index=False)

    def save_word_error_rate(self, ds_id, epoche, wer):
        git_directory = self.get_ds_git_directory(ds_id)

        ds_word_error_rate = {
            'trained_epochs' : epoche,
            'ds_id' : ds_id,
            'wer' : wer
        }
        
        if git_directory:
            with open(f'{git_directory}/wer.json', 'a') as wer_file:
                # wer_file.write(f'{self.trained_epochs:05d} - {ds_id} - WER: {wer:3.4f}\n')
                json.dump(ds_word_error_rate, wer_file)
        else:
            mp3_dir = self.get_snippet_directory(ds_id)
            with open(f'{mp3_dir}/wer.json', 'a') as wer_file:
                # wer_file.write(f'{self.trained_epochs:05d} - {ds_id} - WER: {wer:3.4f}\n')
                json.dump(ds_word_error_rate, wer_file)

    ## return a dict { 'trained_epochs', 'ds_id', 'wer' }
    def get_word_error_rate(self, ds_id):
        git_directory = self.get_ds_git_directory(ds_id)
        
        if git_directory and os.path.isfile(f'{git_directory}/wer.json'):
            with open(f'{git_directory}/wer.json', 'r') as wer_file:
                return json.load(wer_file)

        else:
            mp3_dir = self.get_snippet_directory(ds_id)
            
            if os.path.isfile(f'{mp3_dir}/wer.json'):
                with open(f'{mp3_dir}/wer.json', 'r') as wer_file:
                    return json.load(wer_file)
        
        return {
            'trained_epochs' : 0,
            'ds_id' : ds_id,
            'wer' : 1
        }

    def load_ds_content(self, id_or_directory):
        # Action: translate -> find original
        return self._get_dataframe(id_or_directory, 'content.csv')

    def load_ds_content_with_original(self, id_or_directory):
        # Action: translate -> train
        return self._get_dataframe(id_or_directory, 'content-with_original.csv')

    def load_ds_content_translated(self, id_or_directory):
        # File for Action: find original
        return self._get_dataframe(id_or_directory, 'content-translated.csv')

    def load_ds_content_translated_with_original(self, id_or_directory):
        # TODO: wenn hier ein Verzeichnis stimmt der Algorithmus noch nicht
        # File for Action: train or repeated translation
        git_directory = self.get_ds_git_directory(id_or_directory)
        
        if not git_directory:
            return self._get_dataframe(git_directory, 'content-translated-with_original.csv')
        
        return self._get_dataframe(id_or_directory, 'content-translated-with_original.csv')

    def _get_directory(self, id_or_directory):
        if id_or_directory in self.local_datasets:
            return self.get_snippet_directory(id_or_directory)

        return id_or_directory

    def _get_dataframe(self, id_or_directory, file_name):
        print('-----------------------------')
        print(f'Loading Dataset: {id_or_directory} - {file_name}')
        self.use_dataset(id_or_directory)
        ds_directory = self._get_directory(id_or_directory)
        pandas_df = pd.read_csv(f'{ds_directory}/{file_name}', sep=';')
        truncated_ds = pandas_df

        if 'OriginalText' in pandas_df.columns:
            print(f'Pruning Dataset {id_or_directory} with {pandas_df.shape[0]} Entries')

            if 'Length' in pandas_df.columns:
                truncated_ds = truncated_ds[(truncated_ds.Length <= 4000) & (truncated_ds.Length >= 31)]
                print(f' - {truncated_ds.shape[0]} Entries left after Length Cut (min=31, max=4000)')

            if 'Action' in pandas_df.columns:
                truncated_ds = truncated_ds[~truncated_ds.Action.str.startswith('exclude')]
                print(f' - {truncated_ds.shape[0]} Entries left after Action Cut')
        else:
            if pandas_df.Length.max() > (1600 * 3):
                raise ValueError

        if pandas_df.shape[0] != truncated_ds.shape[0]:
            print(f'Dataset was truncated from {pandas_df.shape[0]} to {truncated_ds.shape[0]} Entries. Saving Backup.')
            pandas_df.to_csv(f'{ds_directory}/original-{file_name}', sep=';', index=False)
            truncated_ds.to_csv(f'{ds_directory}/{file_name}', sep=';', index=False)
        
        pandas_df = truncated_ds
        return pandas_df


def calc_wer(ds_with_translation_and_original, use_akt_translation=True, chunk_size=1000):
    if use_akt_translation and ('Translated1' in ds_with_translation_and_original.columns):
        translation_column = ds_with_translation_and_original.Translated1
    else:
        translation_column = ds_with_translation_and_original.Translated0

    return chunked_wer(
        targets=ds_with_translation_and_original.OriginalText.tolist(),
        predictions=translation_column.tolist(),
        chunk_size=chunk_size
    )

# Chunked version, see https://discuss.huggingface.co/t/spanish-asr-fine-tuning-wav2vec2/4586/5:
def chunked_wer(targets, predictions, chunk_size=1000):
    if chunk_size is None:
        return jiwer.wer(targets, predictions)

    start = 0
    end = chunk_size
    H, S, D, I = 0, 0, 0, 0

    while start < len(targets):
        chunk_metrics = jiwer.compute_measures(targets[start:end], predictions[start:end])
        H = H + chunk_metrics["hits"]
        S = S + chunk_metrics["substitutions"]
        D = D + chunk_metrics["deletions"]
        I = I + chunk_metrics["insertions"]
        start += chunk_size
        end += chunk_size

    return float(S + D + I) / float(H + S + D)
