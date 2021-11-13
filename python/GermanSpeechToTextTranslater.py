import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Union

import torch

from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
from transformers import Trainer, TrainingArguments
from transformers.trainer_pt_utils import LengthGroupedSampler, DistributedLengthGroupedSampler

import json
import math
import collections
import numpy as np
import sklearn
import jiwer
import random
import pandas as pd
from datasets import load_metric
from tqdm.notebook import tqdm_notebook
from sklearn.model_selection import train_test_split
from pathlib import Path
from dictator.SnippetDatasets import SnippetDatasets, calc_wer
from GermanSpeechToTextTranslaterBase import GermanSpeechToTextTranslaterBase


class GermanSpeechToTextTranslater(GermanSpeechToTextTranslaterBase):
    def __init__(
            self,
            model=None,
            processor=None,
            model_name=None,
            ds_handler=None,
            cache_directory=None,
            device='cuda'
    ):
        super().__init__(
            model=model,
            processor=processor,
            model_name=model_name,
            cache_directory=cache_directory,
            device=device
        )
        print('Set Dataset-Handler')
        self.ds_handler = ds_handler if ds_handler else SnippetDatasets()
        print('Loading metric')
        self.my_metric = load_metric('wer')

    def reload_from_checkpoint(self, checkpoint):
        del self.my_model
        torch.cuda.empty_cache()
        self.model_name = checkpoint
        print(f'Using Model: {self.model_name}')
        self.my_model = Wav2Vec2ForCTC.from_pretrained(self.model_name).to(self.device)
        torch.cuda.empty_cache()
        self.my_model.freeze_feature_extractor()

    def translate_and_extend_dataset_from_directory(self, ds_id):
        if not self.ds_handler.needs_translation(ds_id):
            return

        print(f'Translating and extend Dataset: {ds_id}')
        has_original = self.ds_handler.has_content_original(ds_id)

        if not has_original:
            ds = self.ds_handler.load_ds_content(ds_id)
        else:
            ds = self.ds_handler.load_ds_content_with_original(ds_id)

        ds_dir_name = self.ds_handler.get_snippet_directory(ds_id)
        translated_list, size_list = self.translate_dataset(ds_dir_name, ds, ds_id)

        if 'Translated1' in ds.columns:
            ds['Size'] = size_list
            ds['Translated0'] = translated_list
            del ds['Translated1']
        else:
            ds['Size'] = size_list
            ds['Translated0'] = translated_list

        if not has_original:
            ds.to_csv(f'{ds_dir_name}/content-translated.csv', sep=';', index=False)
        else:
            ds.to_csv(f'{ds_dir_name}/content-translated-with_original.csv', sep=';', index=False)

    def translate_dataset(self, mp3_dir, ds, ds_id=None):
        if isinstance(ds, GermanTrainingWav2Vec2Dataset):
            files = [f'{mp3_dir}/{file_name}' for file_name in ds.paths]
            return self.translate_audio_files(files, ds_id)

        return super().translate_dataset(mp3_dir, ds, ds_id)

    def get_trainer(
            self,
            training_args,
            snippet_directory,
            ds_id,
            train_ds,
            test_ds=pd.DataFrame(),
            use_grouped_legth_trainer=False
    ):
        train_dataset = GermanTrainingWav2Vec2Dataset(self, snippet_directory, ds_id, train_ds, 'train')
        test_dataset = GermanTrainingWav2Vec2Dataset(self, snippet_directory, ds_id, test_ds, 'eval')
        data_collator = DataCollatorCTCWithPadding(processor=self.my_processor, padding=True)

        if use_grouped_legth_trainer:
            trainer = GroupedLengthsTrainer(
                model=self.my_model,
                data_collator=data_collator,
                args=training_args,
                compute_metrics=self.compute_metrics,
                train_dataset=train_dataset,
                eval_dataset=test_dataset if not test_ds.empty else None,
                tokenizer=self.my_processor.feature_extractor,
                train_seq_lengths=train_dataset.input_seq_lengths
            )
        else:
            trainer = Trainer(
                model=self.my_model,
                data_collator=data_collator,
                args=training_args,
                compute_metrics=self.compute_metrics,
                train_dataset=train_dataset,
                eval_dataset=test_dataset if not test_ds.empty else None,
                tokenizer=self.my_processor.feature_extractor,
            )

        return trainer

    # returns
    # 1. bad_translated as pandas_df
    # 2. word_error_rate of the hole pandas_df, if diff_calc_wer=True else 1
    def test(self, ds_id, pandas_df=pd.DataFrame(), diff_file_extension=None, diff_calc_wer=True):
        if pandas_df.empty:
            print(f'Loading Dataset: {ds_id}')
            pandas_df = self.ds_handler.load_ds_content_translated_with_original(ds_id)

        translation_column_name = 'Translated1' if 'Translated1' in pandas_df.columns else 'Translated0'

        if not 'OriginalText' in pandas_df.columns:
            raise ValueError

        old_word_error_rate = self.ds_handler.get_word_error_rate(ds_id)
        print(f'aktual trained epoches: {self.trained_epochs}')
        print(f'old trained epoches: {old_word_error_rate["trained_epochs"]}')
        print(f'old word error rate: {old_word_error_rate["wer"]}%')
        bad_translation_ds = pandas_df[pandas_df[translation_column_name] != pandas_df['OriginalText']]
        print(f'No. of bad translated snippets: {bad_translation_ds.shape[0]}')

        if self.trained_epochs == old_word_error_rate['trained_epochs']:
            print('Translation is up to date')
            return pandas_df[pandas_df[translation_column_name] != pandas_df['OriginalText']], old_word_error_rate['wer']/100
        elif old_word_error_rate['trained_epochs'] == 0:
            wer = 100
            print(f'Saving word_error_rate: {wer}%')
            self.ds_handler.save_word_error_rate(ds_id, 0, wer)

        diff_file_extension = diff_file_extension if diff_file_extension else f'{self.trained_epochs:05d}'
        mp3_dir = self.ds_handler.get_snippet_directory(ds_id)

        wer_result = 1.0
        orig_good_translation_ds = pandas_df[pandas_df[translation_column_name] != pandas_df['OriginalText']]
        orig_bad_translation_ds = pandas_df[pandas_df[translation_column_name] == pandas_df['OriginalText']]

        print(f'Translate all')
        predictions, _ = self.translate_dataset(mp3_dir, pandas_df, ds_id)
        pandas_df[translation_column_name] = predictions
        self.ds_handler.save_content_translated_with_original(ds_id, pandas_df, self.trained_epochs)

        print('Calculate WER')
        wer_result = calc_wer(pandas_df)
        wer = 100 * wer_result
        self.ds_handler.save_word_error_rate(ds_id, self.trained_epochs, wer)
        print(f'WER: {wer:3.4f}%')

        good_translation_ds = pandas_df[pandas_df[translation_column_name] == pandas_df['OriginalText']]
        bad_translation_ds = pandas_df[pandas_df[translation_column_name] != pandas_df['OriginalText']]
        print(f'No. of bad translated snippets: {bad_translation_ds.shape[0]}')

        translations = bad_translation_ds[translation_column_name].tolist()
        original_texts = bad_translation_ds['OriginalText'].tolist()
        file_names = bad_translation_ds['Datei'].tolist()

        print('Saving diff files')
        if self.trained_model_directory:
            with open(f'{self.trained_model_directory}/test.log', 'a') as log_file:
                log_file.write(f'{self.trained_epochs:05d} - {ds_id} - WER: {wer:3.4f}\n')

            orig_file = open(f'{self.trained_model_directory}/{ds_id}-original-{diff_file_extension}.txt', 'w')
            translated_file = open(f'{self.trained_model_directory}/{ds_id}-translated-{diff_file_extension}.txt', 'w')

            for file_name, original_text, translation in zip(file_names, original_texts, translations):
                orig_file.write(f'{file_name}, {original_text}\n')
                translated_file.write(f'{file_name}, {translation}\n')

            orig_file.close()
            translated_file.close()

        return bad_translation_ds, wer_result

    def train(
        self,
        trained_model_path,
        dataset_loader,
        ds_to_train,
        max_training_sample_size,
        max_trainingset_size,
        max_rounds,
        num_train_epochs,
        num_steps_per_epoche,  # max_steps,
        per_device_train_batch_size,
        gradient_accumulation_steps,
        logging_steps,
        learning_rate,
        warmup_steps,
        early_stopping_value=0.1
    ):
        manuell_validated_train_actions = ['train7', 'train8', 'train9']

        for runde in range(max_rounds):
            print('======================================')
            print(f'Starting round {runde + 1} of {max_rounds}')
            print('======================================')
            shuffled_ds_to_train = list(ds_to_train)
            random.shuffle(shuffled_ds_to_train)

            for ds_id in shuffled_ds_to_train:
                print('--------------------------------------')
                print(f'-- Start train of Dataset: {ds_id}')
                print('--------------------------------------')
                os.environ['WANDB_NOTES'] = ds_id
                pandas_df = dataset_loader.load_ds_content_translated_with_original(ds_id)
                if not 'OriginalText' in pandas_df.columns:
                    print(f'unable to handle: {ds_id}, cause no Original found')
                    ds_to_train.remove(ds_id)
                    continue

                pandas_df = pandas_df[~pandas_df.Action.str.startswith('validate')]
                print(f'Dataset - {ds_id} loaded with {pandas_df.shape[0]} Entries')
                mp3_dir = dataset_loader.get_snippet_directory(ds_id)

                for epoche in range(num_train_epochs):
                    print('**************************************')
                    print(f'Starting round {runde + 1} of {max_rounds}, epoche {epoche + 1} of {num_train_epochs}')
                    print(f'Splitting Dataset {ds_id} with {pandas_df.shape[0]} Entries')
                    bad_translation_ds, wer_result = self.test(ds_id, pandas_df)
                    print(f'Actual number of bad translated {bad_translation_ds.shape[0]}')
                    print(f'Actual WER: {100 * wer_result:3.4f}%')
                    early_stopping = False
                    # validated_ds = bad_translation_ds[bad_translation_ds['Action']].isin(manuell_validated_train_actions)
                    # (bad_translation_ds.shape[0] > min(20, 200 - validated_ds.shape[0]))

                    if (bad_translation_ds.shape[0] > 50) or (wer_result > early_stopping_value):
                        print('prepare training dataset')
                        train_pandas_ds = sklearn.utils.shuffle(bad_translation_ds)

                        max_ts_sample_size = max_training_sample_size

                        if bad_translation_ds.shape[0] > 600:
                            max_ts_sample_size = 140

                        train_pandas_ds = train_pandas_ds[(train_pandas_ds.Length <= max_ts_sample_size) & (train_pandas_ds.Length >= 31)]
                        print(f' - {train_pandas_ds.shape[0]} Entries left after Length Cut (min=31, max={max_ts_sample_size})')

                        train_pandas_ds = train_pandas_ds[~train_pandas_ds.Action.str.startswith('ignore')]
                        print(f' - {train_pandas_ds.shape[0]} Entries left after ignore Cut')

                        if max_trainingset_size:
                            train_pandas_ds = train_pandas_ds[:min(train_pandas_ds.shape[0], max_trainingset_size)]
                            print(f' - {train_pandas_ds.shape[0]} left after Entries Max Samples Cut (max={max_trainingset_size})')

                        training_args = TrainingArguments(
                            output_dir=trained_model_path,
                            group_by_length=True,
                            per_device_train_batch_size=per_device_train_batch_size,
                            per_device_eval_batch_size=per_device_train_batch_size // 2,
                            gradient_accumulation_steps=gradient_accumulation_steps,
                            evaluation_strategy="steps",
                            max_steps=num_steps_per_epoche,
                            fp16=True,
                            # save_steps=save_steps,
                            eval_steps=logging_steps,  # 4 * num_steps_per_epoche,  # eval_steps=eval_steps,
                            logging_steps=logging_steps,
                            learning_rate=learning_rate/max(4, math.log10(wer_result)**2),
                            warmup_steps=warmup_steps,
                            # save_total_limit=2
                        )

                        print(f'Creating Trainer for {ds_id}')
                        trainer = self.get_trainer(
                            training_args,
                            mp3_dir,
                            ds_id,
                            train_pandas_ds,
                            pandas_df.sort_values(by=['Size'], ascending=False).head(20),
                            use_grouped_legth_trainer=False
                        )
                        print(f'Training of Dataset: {ds_id}')
                        train_result = trainer.train()

                        print(f'Save Model')
                        trainer.save_model()
                        metrics = train_result.metrics
                        max_train_samples = train_pandas_ds.shape[0]
                        metrics["train_samples"] = min(max_train_samples, train_pandas_ds.shape[0])
                        trainer.log_metrics("train", metrics)
                        trainer.save_metrics("train", metrics)
                        trainer.save_state()
                        del trainer
                        torch.cuda.empty_cache()
                        self.trained_epochs = self.trained_epochs + 1
                        with open(f'{self.model_name}/trained_epochs.json', 'w') as json_file:
                            json.dump({'trained_epochs': self.trained_epochs}, json_file)
                    else:
                        # mindestens 98% der Sätze wurde korrekt übersetzt. Überprüfung der Problemfälle ist angebracht.
                        # Es hat sich gezeigt, dass das Ergebnis wieder schlechter werden kann.
                        print(f'Early stopping: {ds_id}')
                        print(f'Actual number of bad translated {bad_translation_ds.shape[0]}')
                        print(f'Actual WER: {100 * wer_result:3.4f}')
                        early_stopping = True
                        # keine gute Idee: ds_to_train.remove(ds_id)
                        break

                if not early_stopping:
                    print(f'final check und update of {ds_id}')
                    bad_translation_ds, wer_result = self.test(ds_id, pandas_df)
                    print(f'Actual number of bad translated {bad_translation_ds.shape[0]}')
                    print(f'Actual WER: {100 * wer_result:3.4f}')

                print(f'finished training of {ds_id} on epoche {self.trained_epochs}')

        print('Training finisched!')

    def compute_metrics(self, pred):
        # we do not want to group tokens when computing the metrics
        label_str = self.my_processor.batch_decode(pred.label_ids, group_tokens=False)
        pred_logits = pred.predictions
        pred_ids = np.argmax(pred_logits, axis=-1)
        pred.label_ids[pred.label_ids == -100] = self.my_processor.tokenizer.pad_token_id
        pred_str = self.my_processor.batch_decode(pred_ids)

        # Bugs: strings in the list of strings 'pred_str' have 2 extra characters ' ' and first character in vocab_dict
        #       strings in the list of strings 'label_str' have extra charecters '[UNK]'
        # Correction:
        # pred_str = [item[:-2] for item in pred_str]
        def g(s):
            s = s.strip()

            while s[-5:] == "<unk>":
                s = s[:-5]

            return s

        label_str_c = [g(item) for item in label_str]

        pred_str_c = pred_str
        return {"wer": jiwer.compute_measures(label_str_c, pred_str_c)["wer"]}


class GermanTrainingWav2Vec2Dataset(torch.utils.data.Dataset):
    def __init__(self, german_speech_translator, snippet_directory, ds_id, ds, split):
        super().__init__()
        assert split in {'train', 'eval'}
        self.split = split
        self.snippet_directory = snippet_directory
        self.german_speech_translator = german_speech_translator
        self.max_input_length_quantile = .98
        self.max_input_length = None
        self.ds_id = ds_id

        if split == 'train':
            self.input_seq_lengths = ds['Size'].tolist()
            self.max_input_length = torch.tensor(self.input_seq_lengths).float() \
                .quantile(self.max_input_length_quantile).int().item()

        if not ds.empty:
            self.labels = ds['OriginalText'].tolist()
            self.paths = ds['Datei'].tolist()
        else:
            self.labels = None
            self.paths = None

    def __len__(self):
        return 0 if not self.paths else len(self.paths)

    def __getitem__(self, idx):
        mp3_file = f'{self.snippet_directory}/{self.paths[idx]}'
        # print(f'Loading MP3: {mp3_file}')
        inputs, _ = self.german_speech_translator.audio_file_to_cuda_inputs(
            mp3_file, self.ds_id
        )

        inputs = inputs.squeeze()
        # print( f'Cuda Inputs created. {type(inputs)}, {inputs.shape}')
        if self.split == 'train':
            inputs = inputs[:self.max_input_length]

        label_str = self.labels[idx]
        processor = self.german_speech_translator.my_processor

        with processor.as_target_processor():
            label = processor(label_str).input_ids

        return {'input_values': inputs, 'labels': label}


@dataclass
class DataCollatorCTCWithPadding:
    """
    Data collator that will dynamically pad the inputs received.
    Args:
        processor (:class:`~transformers.Wav2Vec2Processor`)
            The processor used for proccessing the data.
        padding (:obj:`bool`, :obj:`str` or :class:`~transformers.tokenization_utils_base.PaddingStrategy`, `optional`, defaults to :obj:`True`):
            Select a strategy to pad the returned sequences (according to the model's padding side and padding index)
            among:
            * :obj:`True` or :obj:`'longest'`: Pad to the longest sequence in the batch (or no padding if only a single
              sequence if provided).
            * :obj:`'max_length'`: Pad to a maximum length specified with the argument :obj:`max_length` or to the
              maximum acceptable input length for the model if that argument is not provided.
            * :obj:`False` or :obj:`'do_not_pad'` (default): No padding (i.e., can output a batch with sequences of
              different lengths).
        max_length (:obj:`int`, `optional`):
            Maximum length of the ``input_values`` of the returned list and optionally padding length (see above).
        max_length_labels (:obj:`int`, `optional`):
            Maximum length of the ``labels`` returned list and optionally padding length (see above).
        pad_to_multiple_of (:obj:`int`, `optional`):
            If set will pad the sequence to a multiple of the provided value.
            This is especially useful to enable the use of Tensor Cores on NVIDIA hardware with compute capability >=
            7.5 (Volta).
    """
    processor: Wav2Vec2Processor
    padding: Union[bool, str] = True
    max_length: Optional[int] = None
    max_length_labels: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    pad_to_multiple_of_labels: Optional[int] = None

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        # split inputs and labels since they have to be of different lenghts and need
        # different padding methods
        input_features = [{"input_values": feature["input_values"]} for feature in features]
        label_features = [{"input_ids": feature["labels"]} for feature in features]

        # print(f'Padding input: {self.padding}')
        batch = self.processor.pad(
            input_features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )
        with self.processor.as_target_processor():
            labels_batch = self.processor.pad(
                label_features,
                padding=self.padding,
                max_length=self.max_length_labels,
                pad_to_multiple_of=self.pad_to_multiple_of_labels,
                return_tensors="pt",
            )

        # replace padding with -100 to ignore loss correctly
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)
        batch["labels"] = labels

        return batch


# solution from https://discuss.huggingface.co/t/spanish-asr-fine-tuning-wav2vec2/4586/6
class GroupedLengthsTrainer(Trainer):
    # length_field_name should possibly be part of TrainingArguments instead
    def __init__(self, train_seq_lengths: List[int], *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.train_seq_lengths = train_seq_lengths

    def _get_train_sampler(self) -> Optional[torch.utils.data.sampler.Sampler]:
        if isinstance(self.train_dataset, torch.utils.data.IterableDataset) or not isinstance(
                self.train_dataset, collections.abc.Sized
        ):
            return None

        # Build the sampler.
        if self.args.group_by_length:
            # lengths = self.train_dataset[self.length_field_name] if self.length_field_name is not None else None
            model_input_name = self.tokenizer.model_input_names[0] if self.tokenizer is not None else None

            if self.args.world_size <= 1:
                print('Using LengthGroupedSampler')
                return LengthGroupedSampler(
                    self.train_dataset, self.args.train_batch_size, lengths=self.train_seq_lengths,
                    model_input_name=model_input_name
                )
            else:
                print('Using DistributedLengthGroupedSampler')
                return DistributedLengthGroupedSampler(
                    self.train_dataset,
                    self.args.train_batch_size,
                    num_replicas=self.args.world_size,
                    rank=self.args.process_index,
                    lengths=self.train_seq_lengths,
                    model_input_name=model_input_name,
                )

        else:
            return super()._get_train_sampler()
