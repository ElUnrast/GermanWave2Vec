import os
import json
import torch
import librosa
import numpy as np
from tqdm.notebook import tqdm_notebook
from pathlib import Path
from SrcAudioTools import load_mp3_as_sr16000
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor, Wav2Vec2CTCTokenizer, Wav2Vec2FeatureExtractor


class GermanSpeechToTextTranslaterBase:
    def __init__(
            self,
            model=None,
            processor=None,
            model_name=None,
            cache_directory=None,
            default_model_name='facebook/wav2vec2-large-xlsr-53-german',
            device='cuda'
    ):
        self.device = device
        self.cache_directory = cache_directory
        self.model_name = model_name if model_name else default_model_name
        self.trained_model_directory = None
        print(f'Using Model: {self.model_name}')
        print('Loading processor')
        # Anlegen eines eigenen Processors, da in Wav2Vec2Processor.from_pretrained(facebook/wav2vec2-large-xlsr-53-german)
        # kein ß enthalten ist
        self.my_processor = processor if processor else self.create_processor()
        self.trained_epochs = 1

        if os.path.isfile(f'{self.model_name}/pytorch_model.bin'):
            self.trained_model_directory = self.model_name

            if os.path.isfile(f'{self.trained_model_directory}/trained_epochs.json'):
                with open(f'{self.trained_model_directory}/trained_epochs.json', 'r') as json_file:
                    file_contend_json = json.load(json_file)
                    print(f'json loaded: {file_contend_json}')
                    saved_epoche = file_contend_json['trained_epochs']
                    print(f'Saved Epoch: {saved_epoche}')
                    self.trained_epochs = saved_epoche if saved_epoche > 1 else 1
            else:
                with open(f'{self.trained_model_directory}/trained_epochs.json', 'w') as json_file:
                    json.dump({'trained_epochs': self.trained_epochs}, json_file)

        # TODO: in abgeleitete Klasse verlagern
        # print('Loading language tool')
        # self.my_tool = language_tool if language_tool else language_tool_python.LanguageTool('de-DE')
        print(f'Loading model. Epoche {self.trained_epochs}')

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

    def load_as_sr16000(self, audio_file_name):
        if audio_file_name.endswith('.mp3'):
            samples, _ = load_mp3_as_sr16000(audio_file_name)
        else:
            samples, _ = librosa.load(audio_file_name, sr=16_000)  # Downsample to 16kHz
            samples = np.asarray(samples[0])

        return samples, samples.shape[0]

    def audio_to_cuda_inputs(self, numpy_audio: np.ndarray):
        return self.my_processor(numpy_audio, return_tensors="pt", sampling_rate=16_000).input_values

    def audio_file_to_cuda_inputs(self, audio_file_name, ds_id=None):
        if ds_id:
            if self.cache_directory:
                tmp_directory = f'{self.cache_directory}/{ds_id}'

                if not os.path.exists(tmp_directory):
                    print(f'Creating cache directory: {tmp_directory}')
                    os.makedirs(tmp_directory)

                if not os.path.exists(tmp_directory):
                    raise ValueError

                cache_file_name = f'{tmp_directory}/{Path(audio_file_name).name}.cache'

                if os.path.isfile(cache_file_name):
                    db = torch.load(cache_file_name)
                    return db['ci'], db['ss'].item()

        samples, samples_size = self.load_as_sr16000(audio_file_name)
        ci = self.my_processor(samples)

        if ds_id and self.cache_directory:
            db = {'ci': ci, 'ss': torch.tensor([samples_size])}
            torch.save(db, cache_file_name)

        return ci, samples_size

    def translate_numpy_audio(self, numpy_audio: np.ndarray):
        samples = self.audio_to_cuda_inputs(numpy_audio)
        with torch.no_grad():
            logits = self.my_model(samples.to(self.device)).logits

        # Storing predicted ids
        predicted_ids = torch.argmax(logits, dim=-1)
        # Converting audio to text - Passing the prediction to the tokenzer decode to get the transcription
        return self.my_processor.decode(predicted_ids[0]), numpy_audio.shape[0]

    def translate_audio(self, audio_file_name, ds_id=None):
        return self.translate_numpy_audio(self.audio_file_to_cuda_inputs(audio_file_name, ds_id))

    def translate_dataset(self, mp3_dir, ds, cache_id=None):
        files = [f'{mp3_dir}/{file_name}' for file_name in ds.Datei]
        return self.translate_audio_files(files, cache_id)

    def translate_audio_files(self, files, cache_id=None):
        translated_list = []
        size_list = []
        idx = 0

        for file_name in tqdm_notebook(files):
            if (idx % 20) == 0:
                torch.cuda.empty_cache()

            idx = idx + 1
            translated, sample_size = self.translate_audio(file_name, cache_id)
            translated_list.append(translated)
            size_list.append(sample_size)

        return translated_list, size_list

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
