import os
import glob
import pandas as pd
from zipfile import ZipFile


# Known Snippet Directories with content.csv, content-with_original.csv
class SnippetDatasets:
    def __init__(self, run_on_colab, local_audio_base_dir, extern_audio_base_dir=None):
        self.local_datasets = {}
        self.extern_datasets = {}
        self.used_datasets = []
        self.local_audio_base_dir = local_audio_base_dir
        self.extern_audio_base_dir = extern_audio_base_dir

        for dir in self.directories_with_content(local_audio_base_dir):
            self.local_datasets[os.path.basename(dir)] = dir

        if extern_audio_base_dir:
            for zip_file in glob.glob(f'{extern_audio_base_dir}/*.zip'):
                self.extern_datasets[zip_file[0:-4]] = zip_file

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
            zip_file_name = f'{extern_audio_base_dir}/{ds_id_to_use}.zip'
            print(f'Download and extract {zip_file_name} from gdrive')

            with ZipFile(zip_file_name, 'r') as zip:
                zip.extractall(self.local_audio_base_dir)

            self.local_datasets[f'{self.local_audio_base_dir}/{ds_id_to_use}'] = ds_id_to_use
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
        # File for Action: train
        return self._get_dataframe(id_or_directory, 'content-translated-with_original.csv')
    
    def _get_directory(self, id_or_directory):
        if id_or_directory in self.local_datasets:
            return self.get_snippet_directory(id_or_directory)
        
        return id_or_directory
    
    def _get_dataframe(self, id_or_directory, file_name):
        print(f'Loading Dataset: {id_or_directory} - {file_name}')
        self.use_dataset(id_or_directory)
        ds_directory = self._get_directory(id_or_directory)
        pandas_df = pd.read_csv(f'{ds_directory}/{file_name}', sep=';')
        truncated_ds = pandas_df
        
        if 'OriginalText' in pandas_df.columns:
            print(f'Pruning Dataset {id_or_directory} with {pandas_df.shape[0]} Entries')

            if 'Length' in pandas_df.columns:
                truncated_ds = truncated_ds[(truncated_ds.Length <= 4000) & (truncated_ds.Length >= 31)]
                print(f' - {truncated_ds.shape[0]} Entries left after Length Cut (min=31, max={max_sample_size})')

            if 'Action' in pandas_df.columns:
                truncated_ds = truncated_ds[~truncated_ds.Action.str.startswith('exclude')]
                print(f' - {truncated_ds.shape[0]} Entries left after Action Cut')
        else:
            if pandas_df.Length.max() > (1600 * 3):
                raise ValueError
            
        if pandas_df.shape[0] != truncated_ds.shape[0]:
            print(f'Dataset was truncated from {pandas_df.shape[0]} to {truncated_ds.shape[0]} Entries. Saving Backup.')
            pandas_df.to_csv(f'{ds_directory}/original-{file_name}', sep=';')
            truncated_ds.to_csv(f'{ds_directory}/{file_name}', sep=';')
        
        pandas_df = truncated_ds
        return pandas_df
