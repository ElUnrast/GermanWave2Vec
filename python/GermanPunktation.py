import os
import re
import phonetics
import random
import glob
import numpy as np
import pandas as pd
import torch
from transformers import (
    AdamW,
    T5ForConditionalGeneration,
    MT5ForConditionalGeneration,
    PreTrainedTokenizer,
    T5TokenizerFast as T5Tokenizer,
    MT5TokenizerFast as MT5Tokenizer,
    ByT5Tokenizer,
)
import pytorch_lightning as pl
from transformers import AutoTokenizer
from fastT5 import export_and_get_onnx_model
from torch.utils.data import Dataset, DataLoader
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from sklearn.model_selection import train_test_split

'''
# import
from simplet5 import SimpleT5

# instantiate
model = SimpleT5()

# load (supports t5, mt5, byt5)
# flozi00/byt5-german-grammar
model.from_pretrained("t5","t5-base")

# train
model.train(train_df=train_df, # pandas dataframe with 2 columns: source_text & target_text
            eval_df=eval_df, # pandas dataframe with 2 columns: source_text & target_text
            source_max_token_len = 512,
            target_max_token_len = 128,
            batch_size = 8,
            max_epochs = 5,
            use_gpu = True,
            outputdir = "outputs",
            early_stopping_patience_epochs = 0,
            precision = 32
            )

# load trained T5 model
model.load_model("t5","path/to/trained/model/directory", use_gpu=False)

# predict
model.predict("input text for prediction")

# need faster inference on CPU, get ONNX support
model.convert_and_load_onnx_model("path/to/T5 model/directory")
model.onnx_predict("input text for prediction")
'''


class GermanPunctation:
    def __init__(self, model_name='flozi00/byt5-german-grammar', git_repository=None, max_token_length=128):
        self.git_repository = git_repository
        self.chars_to_ignore_regex = "[^A-Za-z0-9\ö\ä\ü\Ö\Ä\Ü\ß\-,;.:?! ']+"
        self.broken_chars_to_ignore_regex = "[^A-Za-z0-9\ö\ä\ü\Ö\Ä\Ü\ß\- ]+"
        self.trained_model_dir = model_name
        self.max_token_length = max_token_length

        if os.path.isfile(f'{self.trained_model_dir}/pytorch_model.bin'):
            print(f'Loading local Model: {self.trained_model_dir}')
            self.tokenizer, self.model = self.load_model("byt5", self.trained_model_dir, use_gpu=self.use_gpu)
        else:
            model_name = 'flozi00/byt5-german-grammar'
            self.tokenizer, self.model = self.from_pretrained('byt5', model_name)

        if torch.cuda.is_available():
            self.use_gpu = True
            self.device = torch.device("cuda")
        else:
            self.use_gpu = False
            self.device = torch.device("cpu")

        self.model.to(self.device)
        self.T5Model = LightningModel(tokenizer=self.tokenizer, model=self.model, outputdir=self.trained_model_dir)

    def from_pretrained(self, model_type="t5", model_name="t5-base") -> None:
        """
        loads T5/MT5 Model model for training/finetuning

        Args:
            model_type (str, optional): "t5" or "mt5" . Defaults to "t5".
            model_name (str, optional): exact model architecture name, "t5-base" or "t5-large". Defaults to "t5-base".
        """
        if model_type == "t5":
            return T5Tokenizer.from_pretrained(f"{model_name}"), T5ForConditionalGeneration.from_pretrained(f"{model_name}", return_dict=True)
        elif model_type == "mt5":
            return MT5Tokenizer.from_pretrained(f"{model_name}"), MT5ForConditionalGeneration.from_pretrained(f"{model_name}", return_dict=True)
        elif model_type == "byt5":
            return ByT5Tokenizer.from_pretrained(f"{model_name}"), T5ForConditionalGeneration.from_pretrained(f"{model_name}", return_dict=True)

    def load_model(self, model_type: str = "t5", model_dir: str = "outputs"):
        """
        loads a checkpoint for inferencing/prediction

        Args:
            model_type (str, optional): "t5" or "mt5". Defaults to "t5".
            model_dir (str, optional): path to model directory. Defaults to "outputs".
            use_gpu (bool, optional): if True, model uses gpu for inferencing/prediction. Defaults to True.
        """
        if model_type == "t5":
            return T5Tokenizer.from_pretrained(f"{model_dir}"), T5ForConditionalGeneration.from_pretrained(f"{model_dir}")
        elif model_type == "mt5":
            return MT5Tokenizer.from_pretrained(f"{model_dir}"), MT5ForConditionalGeneration.from_pretrained(f"{model_dir}")
        elif model_type == "byt5":
            return ByT5Tokenizer.from_pretrained(f"{model_dir}"), T5ForConditionalGeneration.from_pretrained(f"{model_dir}")

    def punctate_text(self, text):
        return self.model.predict(text)

    def extract_source_and_target_text(self, text):
        source_text = re.sub(self.chars_to_ignore_regex, '', text)
        targetn_text = re.sub(self.broken_chars_to_ignore_regex, "", text.lower())
        return source_text, targetn_text

    def do_text_manipulation(self, source_text):
        broken_text = source_text

        if(random.randint(0, 100) >= 50):
            for xyz in range(int(len(broken_text.split(" "))/4)):
                if(random.randint(0, 100) > 30):
                    randc = random.choice(broken_text.split(" "))

                    if(random.randint(0, 10) > 4):
                        broken_text = broken_text.replace(randc, ''.join(random.choice('abcdefghijklmnopqrstuvxyz') for _ in range(len(randc))).lower())
                    else:
                        broken_text = broken_text.replace(randc, phonetics.metaphone(randc).lower())

        return broken_text

    def train(
        self,
        train_df: pd.DataFrame,
        eval_df: pd.DataFrame,
        batch_size: int = 8,
        max_epochs: int = 5,
        early_stopping_patience_epochs: int = 0,  # 0 to disable early stopping feature
        precision=32
    ):
        """
        trains T5/MT5 model on custom dataset

        Args:
            train_df (pd.DataFrame): training datarame. Dataframe must have 2 column --> "source_text" and "target_text"
            eval_df ([type], optional): validation datarame. Dataframe must have 2 column --> "source_text" and "target_text"
            batch_size (int, optional): batch size. Defaults to 8.
            max_epochs (int, optional): max number of epochs. Defaults to 5.
            early_stopping_patience_epochs (int, optional): monitors val_loss on epoch end and stops training, if val_loss does not improve after the specied number of epochs. set 0 to disable early stopping. Defaults to 0 (disabled)
            precision (int, optional): sets precision training - Double precision (64), full precision (32) or half precision (16). Defaults to 32.
        """
        data_module = LightningDataModule(
            train_df,
            eval_df,
            self.tokenizer,
            batch_size=batch_size,
            source_max_token_len=self.max_token_length,
            target_max_token_len=self.max_token_length
        )

        early_stop_callback = (
            [
                EarlyStopping(
                    monitor="val_loss",
                    min_delta=0.00,
                    patience=early_stopping_patience_epochs,
                    verbose=True,
                    mode="min",
                )
            ]
            if early_stopping_patience_epochs > 0
            else None
        )

        gpus = 1 if self.use_gpu else 0

        trainer = pl.Trainer(
            callbacks=early_stop_callback,
            max_epochs=max_epochs,
            gpus=gpus,
            progress_bar_refresh_rate=5,
            precision=precision
        )

        trainer.fit(self.T5Model, data_module)
        trainer.optimizers
        return trainer

    def train_potter(self):
        txt_directory = f'{self.git_repository}/datasets/punkctation'
        dataset_file_name = f'{self.git_repository}/datasets/punkctation/potter_punktation.csv'

        if os.path.isfile(dataset_file_name):
            pandas_df = pd.read_csv(dataset_file_name, sep=';')
        else:
            all_potters = ''
            for text_file in glob.glob(f'{txt_directory}/*.txt'):
                first = True

                with open(text_file, encoding='utf8') as f:
                    print(f'reading: {text_file}')
                    for line in f:
                        if not first:
                            all_potters += ' '
                        first = False
                        all_potters += line.strip()

            print(f'all Potters length: {len(all_potters):d}')
            source_text_list = []
            target_text_list = []
            first_index = 0
            last_index = self.max_token_length

            while True:
                last_index = all_potters.rfind(' ', first_index, last_index)
                target_text, source_text = self.extract_source_and_target_text(all_potters[first_index:last_index])
                source_text_list.append(source_text)
                target_text_list.append(target_text)

                first_index = last_index + 1
                last_index += self.max_token_length

                if last_index > len(all_potters):
                    break

            pandas_df = pd.DataFrame()
            pandas_df['source_text'] = source_text_list
            pandas_df['target_text'] = target_text_list

            if self.git_repository:
                pandas_df.to_csv(dataset_file_name, sep=';', index=False)

        pandas_df['target_text'] = pandas_df['target_text'].apply(self.do_text_manipulation)
        train, test = train_test_split(pandas_df, test_size=0.2)

        for i in range(0, len(train) - 100, 100):
            print(f'range start: {i:d}')
            train1 = train.iloc[i:i + 100]
            test1 = test.iloc[i // 5:(i // 5) + 20]
            trainer = self.train(train_df=train1, eval_df=test1, batch_size=4)
            trainer.save_model()
            del trainer
            torch.cuda.empty_cache()

    def predict(
        self,
        source_text: str,
        max_length: int = 512,
        num_return_sequences: int = 1,
        num_beams: int = 2,
        top_k: int = 50,
        top_p: float = 0.95,
        do_sample: bool = True,
        repetition_penalty: float = 2.5,
        length_penalty: float = 1.0,
        early_stopping: bool = True,
        skip_special_tokens: bool = True,
        clean_up_tokenization_spaces: bool = True,
    ):
        """
        generates prediction for T5/MT5 model

        Args:
            source_text (str): any text for generating predictions
            max_length (int, optional): max token length of prediction. Defaults to 512.
            num_return_sequences (int, optional): number of predictions to be returned. Defaults to 1.
            num_beams (int, optional): number of beams. Defaults to 2.
            top_k (int, optional): Defaults to 50.
            top_p (float, optional): Defaults to 0.95.
            do_sample (bool, optional): Defaults to True.
            repetition_penalty (float, optional): Defaults to 2.5.
            length_penalty (float, optional): Defaults to 1.0.
            early_stopping (bool, optional): Defaults to True.
            skip_special_tokens (bool, optional): Defaults to True.
            clean_up_tokenization_spaces (bool, optional): Defaults to True.

        Returns:
            list[str]: returns predictions
        """
        input_ids = self.tokenizer.encode(
            source_text, return_tensors="pt", add_special_tokens=True
        )
        with torch.no_grad():
            input_ids = input_ids.to(self.device)
            generated_ids = self.model.generate(
                input_ids=input_ids,
                num_beams=num_beams,
                max_length=max_length,
                repetition_penalty=repetition_penalty,
                length_penalty=length_penalty,
                early_stopping=early_stopping,
                top_p=top_p,
                top_k=top_k,
                num_return_sequences=num_return_sequences,
            )
            preds = [
                self.tokenizer.decode(
                    g,
                    skip_special_tokens=skip_special_tokens,
                    clean_up_tokenization_spaces=clean_up_tokenization_spaces,
                )
                for g in generated_ids
            ]
            return preds

    def convert_and_load_onnx_model(self, model_dir: str):
        """ returns ONNX model """
        self.onnx_model = export_and_get_onnx_model(model_dir)
        self.onnx_tokenizer = AutoTokenizer.from_pretrained(model_dir)

    def onnx_predict(self, source_text: str):
        """ generates prediction from ONNX model """
        token = self.onnx_tokenizer(source_text, return_tensors="pt")
        tokens = self.onnx_model.generate(
            input_ids=token["input_ids"],
            attention_mask=token["attention_mask"],
            num_beams=2,
        )
        output = self.onnx_tokenizer.decode(tokens.squeeze(), skip_special_tokens=True)
        return output


class PyTorchDataModule(Dataset):
    """  PyTorch Dataset class  """

    def __init__(
        self,
        data: pd.DataFrame,
        tokenizer: PreTrainedTokenizer,
        source_max_token_len: int = 512,
        target_max_token_len: int = 512,
    ):
        """
        initiates a PyTorch Dataset Module for input data

        Args:
            data (pd.DataFrame): input pandas dataframe. Dataframe must have 2 column --> "source_text" and "target_text"
            tokenizer (PreTrainedTokenizer): a PreTrainedTokenizer (T5Tokenizer, MT5Tokenizer, or ByT5Tokenizer)
            source_max_token_len (int, optional): max token length of source text. Defaults to 512.
            target_max_token_len (int, optional): max token length of target text. Defaults to 512.
        """
        self.tokenizer = tokenizer
        self.data = data
        self.source_max_token_len = source_max_token_len
        self.target_max_token_len = target_max_token_len

    def __len__(self):
        """ returns length of data """
        return len(self.data)

    def __getitem__(self, index: int):
        """ returns dictionary of input tensors to feed into T5/MT5 model"""

        data_row = self.data.iloc[index]
        source_text = data_row["source_text"]

        source_text_encoding = self.tokenizer(
            source_text,
            max_length=self.source_max_token_len,
            padding="max_length",
            truncation=True,
            return_attention_mask=True,
            add_special_tokens=True,
            return_tensors="pt",
        )

        target_text_encoding = self.tokenizer(
            data_row["target_text"],
            max_length=self.target_max_token_len,
            padding="max_length",
            truncation=True,
            return_attention_mask=True,
            add_special_tokens=True,
            return_tensors="pt",
        )

        labels = target_text_encoding["input_ids"]
        labels[
            labels == 0
        ] = -100  # to make sure we have correct labels for T5 text generation

        return dict(
            source_text=source_text,
            target_text=data_row["target_text"],
            source_text_input_ids=source_text_encoding["input_ids"].flatten(),
            source_text_attention_mask=source_text_encoding["attention_mask"].flatten(),
            labels=labels.flatten(),
            labels_attention_mask=target_text_encoding["attention_mask"].flatten(),
        )


class LightningDataModule(pl.LightningDataModule):
    """ PyTorch Lightning data class """

    def __init__(
        self,
        train_df: pd.DataFrame,
        test_df: pd.DataFrame,
        tokenizer: PreTrainedTokenizer,
        batch_size: int = 4,
        source_max_token_len: int = 512,
        target_max_token_len: int = 512,
    ):
        """
        initiates a PyTorch Lightning Data Module

        Args:
            train_df (pd.DataFrame): training dataframe. Dataframe must contain 2 columns --> "source_text" & "target_text"
            test_df (pd.DataFrame): validation dataframe. Dataframe must contain 2 columns --> "source_text" & "target_text"
            tokenizer (PreTrainedTokenizer): PreTrainedTokenizer (T5Tokenizer, MT5Tokenizer, or ByT5Tokenizer)
            batch_size (int, optional): batch size. Defaults to 4.
            source_max_token_len (int, optional): max token length of source text. Defaults to 512.
            target_max_token_len (int, optional): max token length of target text. Defaults to 512.
        """
        super().__init__()

        self.train_df = train_df
        self.test_df = test_df
        self.batch_size = batch_size
        self.tokenizer = tokenizer
        self.source_max_token_len = source_max_token_len
        self.target_max_token_len = target_max_token_len

    def setup(self, stage=None):
        self.train_dataset = PyTorchDataModule(
            self.train_df,
            self.tokenizer,
            self.source_max_token_len,
            self.target_max_token_len,
        )
        self.test_dataset = PyTorchDataModule(
            self.test_df,
            self.tokenizer,
            self.source_max_token_len,
            self.target_max_token_len,
        )

    def train_dataloader(self):
        """ training dataloader """
        return DataLoader(
            self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=2
        )

    def test_dataloader(self):
        """ test dataloader """
        return DataLoader(
            self.test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=2
        )

    def val_dataloader(self):
        """ validation dataloader """
        return DataLoader(
            self.test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=2
        )


class LightningModel(pl.LightningModule):
    """ PyTorch Lightning Model class"""

    def __init__(self, tokenizer, model, outputdir: str = "outputs"):
        """
        initiates a PyTorch Lightning Model

        Args:
            tokenizer : T5/MT5/ByT5 tokenizer
            model : T5/MT5/ByT5 model
            outputdir (str, optional): output directory to save model checkpoints. Defaults to "outputs".
        """
        super().__init__()
        self.model = model
        self.tokenizer = tokenizer
        self.outputdir = outputdir

    def forward(self, input_ids, attention_mask, decoder_attention_mask, labels=None):
        """ forward step """
        output = self.model(
            input_ids,
            attention_mask=attention_mask,
            labels=labels,
            decoder_attention_mask=decoder_attention_mask,
        )

        return output.loss, output.logits

    def training_step(self, batch, batch_size):
        """ training step """
        input_ids = batch["source_text_input_ids"]
        attention_mask = batch["source_text_attention_mask"]
        labels = batch["labels"]
        labels_attention_mask = batch["labels_attention_mask"]

        loss, outputs = self(
            input_ids=input_ids,
            attention_mask=attention_mask,
            decoder_attention_mask=labels_attention_mask,
            labels=labels,
        )

        self.log("train_loss", loss, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_size):
        """ validation step """
        input_ids = batch["source_text_input_ids"]
        attention_mask = batch["source_text_attention_mask"]
        labels = batch["labels"]
        labels_attention_mask = batch["labels_attention_mask"]

        loss, outputs = self(
            input_ids=input_ids,
            attention_mask=attention_mask,
            decoder_attention_mask=labels_attention_mask,
            labels=labels,
        )

        self.log("val_loss", loss, prog_bar=True, logger=True)
        return loss

    def test_step(self, batch, batch_size):
        """ test step """
        input_ids = batch["source_text_input_ids"]
        attention_mask = batch["source_text_attention_mask"]
        labels = batch["labels"]
        labels_attention_mask = batch["labels_attention_mask"]

        loss, outputs = self(
            input_ids=input_ids,
            attention_mask=attention_mask,
            decoder_attention_mask=labels_attention_mask,
            labels=labels,
        )

        self.log("test_loss", loss, prog_bar=True, logger=True)
        return loss

    def configure_optimizers(self):
        """ configure optimizers """
        return AdamW(self.parameters(), lr=0.0001)

    def training_epoch_end(self, training_step_outputs):
        """ save tokenizer and model on epoch end """
        avg_traning_loss = np.round(
            torch.mean(torch.stack([x["loss"] for x in training_step_outputs])).item(),
            4,
        )
        # path = f"{self.outputdir}/SimpleT5-epoch-{self.current_epoch}-train-loss-{str(avg_traning_loss)}"
        path = self.outputdir
        # sollte sich nicht verändert haben
        # self.tokenizer.save_pretrained(path)
        print(f'Saving Model to: {path}')
        self.model.save_pretrained(path)

    # def validation_epoch_end(self, validation_step_outputs):
    #     # val_loss = torch.stack([x['loss'] for x in validation_step_outputs]).mean()
    #     path = f"{self.outputdir}/T5-epoch-{self.current_epoch}"
    #     self.tokenizer.save_pretrained(path)
    #     # self.model.save_pretrained(path)


def main():
    punctation = GermanPunctation()
    # ''Ein weiterer lauter Knall ertönte, und Dobby, Luna, Dean und Ollivander verschwanden.''
    print(punctation.punctate_text(f'ein weiterer lauter knall ertönte und dobby luna dean und ollivander verschwanden'))
    print(punctation.punctate_text(f'correct german grammar: ein weiterer lauter knall ertönte und dobby luna dean und ollivander verschwanden'))
    print(punctation.punctate_text(f'es ist schön so viele tolle menschen um sich zu haben denn ohne sie wäre es nicht so schön'))
    print(punctation.punctate_text(f'correct german grammar: es ist schön so viele tolle menschen um sich zu haben denn ohne sie wäre es nicht so schön'))


if __name__ == '__main__':
    main()
