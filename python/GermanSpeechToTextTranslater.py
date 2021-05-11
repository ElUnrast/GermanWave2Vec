import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union

import torch
import torchaudio
from torch import nn
from torch.cuda.amp import autocast

from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor, Wav2Vec2CTCTokenizer, Wav2Vec2FeatureExtractor
from transformers import Trainer, TrainingArguments
from transformers.trainer_utils import get_last_checkpoint, is_main_process
from transformers.trainer_pt_utils import LengthGroupedSampler, DistributedLengthGroupedSampler

import json
import collections
import librosa
import numpy as np
import pandas as pd
import sklearn
import jiwer
from jiwer import wer
from datasets import load_metric
from tqdm.notebook import tqdm_notebook
from sklearn.model_selection import train_test_split


class GermanSpeechToTextTranslater:
    def __init__(
            self,
            model=None,
            processor=None,
            # language_tool=None,
            # resampled_dir=None,
            model_name=None,
            default_model_name='facebook/wav2vec2-large-xlsr-53-german',
            device='cuda'
    ):
        self.device = device
        self.model_name = model_name if model_name else default_model_name
        print(f'Using Model: {self.model_name}')
        print('Loading processor')
        # Anlegen eines eigenen Processors, da in Wav2Vec2Processor.from_pretrained(facebook/wav2vec2-large-xlsr-53-german)
        # kein ß enthalten ist
        self.my_processor = processor if processor else self.create_processor()
        print('Loading metric')
        self.my_metric = load_metric('wer')
        # TODO: in abgeleitete Klasse verlagern
        # print('Loading language tool')
        # self.my_tool = language_tool if language_tool else language_tool_python.LanguageTool('de-DE')
        print('Loading model')

        if model:
            self.my_model = model
        else:
            if model_name:
                self.my_model = Wav2Vec2ForCTC.from_pretrained(self.model_name).to(device)
            else:
                self.my_model = Wav2Vec2ForCTC.from_pretrained(
                    self.model_name,
                    # activation_dropout=0.055
                    attention_dropout=0.1,  # 0.094
                    hidden_dropout=0.1,  #
                    feat_proj_dropout=0.0,  # 0.04
                    mask_time_prob=0.05,  # 0.08
                    layerdrop=0.1,  # 0.04
                    gradient_checkpointing=True,
                    ctc_loss_reduction="mean",
                    pad_token_id=self.my_processor.tokenizer.pad_token_id,
                    vocab_size=len(self.my_processor.tokenizer)
                ).to(device)
        self.my_model.freeze_feature_extractor()

    def reload_from_checkpoint(self, checkpoint):
        del self.my_model
        torch.cuda.empty_cache()
        self.model_name = checkpoint
        print(f'Using Model: {self.model_name}')
        self.my_model = Wav2Vec2ForCTC.from_pretrained(self.model_name).to(device)
        torch.cuda.empty_cache()
        self.my_model.freeze_feature_extractor()

    # def translate(self, audio_file_name):
    #    translation = self.translate_audio(audio_file_name)
    #    return self.my_tool.correct(translation), translation

    def load_as_sr16000(self, audio_file_name):
        new_path = None

        if not audio_file_name.endswith('.mp3'):
            samples, sampling_rate = librosa.load(audio_file_name, sr=16_000)  # Downsample to 16kHz
        else:
            # print( f'load {audio_file_name}')
            samples, sampling_rate = torchaudio.load(audio_file_name)
            # print(f'samples.shape : {samples.shape}, sampling_rate : {sampling_rate}')
            samples = samples[0]

            if sampling_rate != 16_000:
                # print( f'Converting from {sampling_rate}')
                samples = librosa.resample(np.asarray(samples), sampling_rate, 16_000)
                # funktioniert auch:
                # samples = torchaudio.transforms.Resample(orig_freq=sampling_rate, new_freq=16_000)(samples).squeeze().numpy()

        if samples.shape[0] == 2:
            samples = samples[0]

        if isinstance(samples, (np.ndarray, np.generic)):
            samples = torch.from_numpy(samples).float().flatten()
        else:
            samples = samples.squeeze().numpy()

        if new_path != None:
            torch.save(samples, new_path)

        samples_size = samples.shape[0]
        return samples, samples.shape[0]

    def audio_to_cuda_inputs(self, audio_file_name):
        samples, samples_size = self.load_as_sr16000(audio_file_name)
        return self.my_processor(samples, return_tensors="pt", sampling_rate=16_000).input_values, samples_size

    def translate_audio(self, audio_file_name):
        samples, samples_size = self.audio_to_cuda_inputs(audio_file_name)
        samples_size = samples_size

        with torch.no_grad():
            logits = self.my_model(samples.to(self.device)).logits

        # Storing predicted ids
        predicted_ids = torch.argmax(logits, dim=-1)
        # Converting audio to text - Passing the prediction to the tokenzer decode to get the transcription
        return self.my_processor.decode(predicted_ids[0]), samples_size

    def translate_and_extend_dataset_from_directory(self, ds_loader, id_or_directory):
        if not ds_loader.needs_translation(id_or_directory):
            return

        print(f'Translating and extend Dataset: {id_or_directory}')
        has_original = ds_loader.has_content_original(id_or_directory)

        if not has_original:
            ds = ds_loader.load_ds_content(id_or_directory)
        else:
            ds = ds_loader.load_ds_content_with_original(id_or_directory)

        ds_dir_name = ds_loader.get_snippet_directory(id_or_directory)
        translated_list, size_list = self.translate_dataset(ds_dir_name, ds)

        if 'Translated1' in ds.columns:
            ds['Size'] = size_list
            ds['Translated0'] = translated_list
            del ds['Translated1']
        elif 'Translated0' in ds.columns:
            ds['Size'] = size_list
            ds['Translated0'] = translated_list
        else:
            ds.insert(loc=8, column='Size', value=size_list)
            ds.insert(loc=9, column='Translated0', value=translated_list)

        if not has_original:
            ds.to_csv(f'{ds_dir_name}/content-translated.csv', sep=';')
        else:
            ds.to_csv(f'{ds_dir_name}/content-translated-with_original.csv', sep=';')

    def translate_dataset(self, mp3_dir, ds):
        if isinstance(ds, GermanTrainingWav2Vec2Dataset):
            files = [f'{mp3_dir}/{file_name}' for file_name in ds.paths]
        else:
            files = [f'{mp3_dir}/{file_name}' for file_name in ds.Datei]

        return self.translate_audio_files(files)

    def translate_audio_files(self, files):
        translated_list = []
        size_list = []
        idx = 0

        for file_name in tqdm_notebook(files):
            if (idx % 20) == 0:
                torch.cuda.empty_cache()

            idx = idx + 1
            translated, sample_size = self.translate_audio(file_name)
            translated_list.append(translated)
            size_list.append(sample_size)

        return translated_list, size_list

    def split_dataset(
            self,
            pandas_df,
            max_trainingset_size=25000,
            max_sample_size=1000,
            use_only_incorrect_translated=True
    ):
        used_df = pandas_df
        has_action = 'Action' in pandas_df.columns

        if has_action:
            used_df = used_df[(used_df.Action == 'train') | (used_df.Action == 'translate')]
            print(f' - {used_df.shape[0]} Entries left after Action Cut')

        used_df = used_df[used_df.Length <= max_sample_size] if max_sample_size else pandas_df
        print(f' - {used_df.shape[0]} Entries left after Length Cut (max={max_sample_size})')
        used_df = used_df[used_df.Length > 30]
        print(f' - {used_df.shape[0]} Entries left after minimal Size Cut (min=31)')

        # Test Dataset should be fixed_training_set_size Entries long
        test_percentage = fixed_training_set_size / used_df.shape[0] if fixed_training_set_size else 0.2

        train, test = train_test_split(used_df, test_size=test_percentage, random_state=143)
        print(f'Training Dataset Size: {train.shape[0]}, Validation Dataset Size {test.shape[0]}')

        if use_only_incorrect_translated:
            if 'Translated1' in train.columns:
                train = train[(train.OriginalText != train.Translated1)]
            else:
                train = train[(train.OriginalText != train.Translated0)]

            print(f' - {train.shape[0]} Entries left after first Recognition Cut')

        train = sklearn.utils.shuffle(train)

        if max_trainingset_size:
            train = train[:min(train.shape[0], max_trainingset_size)]
            print(f' - {train.shape[0]} left after Entries Max Samples Cut (max={max_trainingset_size})')

        print(f'Training Dataset Size: {train.shape[0]}, Validation Dataset Size {test.shape[0]}')
        return train, test

    def get_trainer(
            self,
            training_args,
            snippet_directory,
            train_ds,
            test_ds,
            use_grouped_legth_trainer=False
    ):
        train_dataset = GermanTrainingWav2Vec2Dataset(self, snippet_directory, train_ds, 'train')
        test_dataset = GermanTrainingWav2Vec2Dataset(self, snippet_directory, test_ds, 'eval')
        data_collator = DataCollatorCTCWithPadding(processor=self.my_processor, padding=True)

        if use_grouped_legth_trainer:
            trainer = GroupedLengthsTrainer(
                model=self.my_model,
                data_collator=data_collator,
                args=training_args,
                compute_metrics=self.compute_metrics,
                train_dataset=train_dataset,
                eval_dataset=test_dataset,
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
                eval_dataset=test_dataset,
                tokenizer=self.my_processor.feature_extractor,
            )

        return trainer

    def create_processor(self):
        vocab_file_name = 'vocab.json'

        if not os.path.isfile(vocab_file_name):
            vocab_dict = {
                '<pad>': 0, '<s>': 1, '</s>': 2, '<unk>': 3, '|': 4,
                'e': 5, 'n': 6, 'i': 7, 's': 8, 'r': 9, 't': 10, 'a': 11,
                'h': 12, 'd': 13, 'u': 14, 'l': 15, 'c': 16, 'g': 17, 'm': 18,
                'o': 19, 'b': 20, 'w': 21, 'f': 22, 'k': 23, 'z': 24, 'v': 25,
                'ü': 26, 'p': 27, 'ä': 28, 'ö': 29, 'j': 30, 'y': 31, "'": 32,
                'x': 33, 'q': 34, 'ß': 35
            }

            with open(vocab_file_name, 'w') as vocab_file:
                json.dump(vocab_dict, vocab_file)

        tokenizer = Wav2Vec2CTCTokenizer(
            "vocab.json",
            unk_token="<unk>",
            pad_token="<pad>",
            word_delimiter_token="|"
        )
        feature_extractor = Wav2Vec2FeatureExtractor(
            feature_size=1, sampling_rate=16_000, padding_value=0.0, do_normalize=True, return_attention_mask=True
        )
        return Wav2Vec2Processor(feature_extractor=feature_extractor, tokenizer=tokenizer)

    def test(self, ds_id, pandas_df, mp3_dir, diff_file_extension, diff_calc_wer=False):
        if not 'Autor' in pandas_df.columns:
            raise ValueError

        wer_result = 1.0
        translation_column_name = 'Translated1'

        print(f'Translate all')
        predictions, _ = self.translate_dataset(mp3_dir, pandas_df)
        pandas_df[translation_column_name] = predictions
        pandas_df.to_csv(f'{mp3_dir}/content-translated-with_original.csv', sep=';')

        if diff_calc_wer:
            print('Calculate WER')
            wer_result = self.calc_wer(pandas_df)
            print(f'WER: {wer_result}')

        print('Saving diff files')
        truncated_ds = pandas_df[
            ((pandas_df.Action == 'train') | (pandas_df.Action == 'translate')) & (pandas_df.Length <= 1200)
        ]
        bad_translation_ds = truncated_ds[truncated_ds[translation_column_name] != truncated_ds['OriginalText']]

        translations = bad_translation_ds[translation_column_name].tolist()
        original_texts = bad_translation_ds['OriginalText'].tolist()
        file_names = bad_translation_ds['Datei'].tolist()

        orig_file = open(f'{model_dir}/{ds_id}-original-{diff_file_extension}.txt', 'w')
        translated_file = open(f'{model_dir}/{ds_id}-translated-{diff_file_extension}.txt', 'w')

        for file_name, original_text, translation in zip(file_names, original_texts, translations):
            orig_file.write(f'{file_name}, {original_text}\n')
            translated_file.write(f'{file_name}, {translation}\n')

        orig_file.close()
        translated_file.close()

        return bad_translation_ds, truncated_ds, wer_result

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

        # def f(s):
        #     if len(s) <= 2:    # maybe 2 -> 1 later
        #         return s
        #     else:
        #         return s[:-2]   # maybe 2 -> 1 later

        # pred_str_c = [f(item) for item in pred_str]
        pred_str_c = pred_str
        # print(f'"{label_str_c}" - "{pred_str_c}"')
        return {"wer": jiwer.compute_measures(label_str_c, pred_str_c)["wer"]}

    def calc_wer(self, ds_with_translation_and_original, chunk_size=1000):
        if 'Translated1' in ds_with_translation_and_original.columns:
            translation_column = ds_with_translation_and_original.Translated1
        else:
            translation_column = ds_with_translation_and_original.Translated0

        return chunked_wer(
            targets=ds_with_translation_and_original.OriginalText,
            predictions=translation_column,
            chunk_size=chunk_size
        )

    # Chunked version, see https://discuss.huggingface.co/t/spanish-asr-fine-tuning-wav2vec2/4586/5:
    def chunked_wer(self, targets, predictions, chunk_size=1000):
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


class GermanTrainingWav2Vec2Dataset(torch.utils.data.Dataset):
    def __init__(self, german_speech_translator, snippet_directory, ds, split):
        super().__init__()
        assert split in {'train', 'eval'}
        self.split = split
        self.snippet_directory = snippet_directory
        self.german_speech_translator = german_speech_translator
        self.max_input_length_quantile = .98
        self.max_input_length = None

        if split == 'train':
            self.input_seq_lengths = ds['Size'].tolist()
            self.max_input_length = torch.tensor(self.input_seq_lengths).float() \
                .quantile(self.max_input_length_quantile).int().item()

        self.labels = ds['OriginalText'].tolist()
        self.paths = ds['Datei'].tolist()

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        mp3_file = f'{self.snippet_directory}/{self.paths[idx]}'
        # print(f'Loading MP3: {mp3_file}')
        inputs, _ = self.german_speech_translator.audio_to_cuda_inputs(
            mp3_file
        )

        inputs = inputs.squeeze()
        # print( f'Cuda Inputs created. {type(inputs)}, {inputs.shape}')
        if self.split == 'train':
            inputs = inputs[:self.max_input_length]

        label_str = self.labels[idx]
        # print(f'Label: {type(label_str)}')

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
