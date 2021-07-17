from simplet5 import SimpleT5
import os
import re
import phonetics
import random
import glob
import pandas as pd
import torch
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
    def __init__(self, model_name='flozi00/byt5-german-grammar', git_repository=None, use_gpu=False):
        self.model = SimpleT5()
        self.git_repository = git_repository
        self.chars_to_ignore_regex = "[^A-Za-z0-9\ö\ä\ü\Ö\Ä\Ü\ß\-,;.:?! ']+"
        self.broken_chars_to_ignore_regex = "[^A-Za-z0-9\ö\ä\ü\Ö\Ä\Ü\ß\- ]+"
        self.trained_model_dir = model_name
        self.use_gpu = use_gpu

        if os.path.isdir(self.trained_model_dir):
            self.model.load_model("byt5", self.trained_model_dir, use_gpu=self.use_gpu)
        else:
            model_name = 'flozi00/byt5-german-grammar'
            self.model.from_pretrained('byt5', model_name)

        if use_gpu:
            if torch.cuda.is_available():
                self.device = torch.device("cuda")
            else:
                raise "exception ---> no gpu found. set use_gpu=False, to use CPU"
        else:
            self.device = torch.device("cpu")

        self.model.device = self.device
        self.model.model.to(self.device)

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
        train_df,  # pandas dataframe with 2 columns: source_text & target_text
        eval_df,  # pandas dataframe with 2 columns: source_text & target_text
        batch_size=8,
        max_epochs=5,
        early_stopping_patience_epochs=0,
        precision=32
    ):
        self.model.train(
            train_df=train_df,
            eval_df=eval_df,
            source_max_token_len=500,
            target_max_token_len=500,
            batch_size=batch_size,
            max_epochs=max_epochs,
            use_gpu=self.use_gpu,
            outputdir=self.trained_model_dir,
            early_stopping_patience_epochs=early_stopping_patience_epochs,
            precision=precision
        )

    def train_potter(self):
        if self.git_repository:
            dataset_file_name = f'{self.git_repository}/datasets/punkctation/potter_punktation.csv'

        if os.path.isfile(dataset_file_name):
            pandas_df = pd.read_csv(dataset_file_name, sep=';')
        else:
            all_potters = ''
            for text_file in glob.glob('C:/share/NLP-Data/*.txt'):
                first = True

                with open('quotes.txt', encoding='utf8') as f:
                    for line in f:
                        if not first:
                            all_potters += ' '
                        first = False
                        all_potters += line.strip()

            first_index = 0
            last_index = 500

            pandas_df = pd.DataFrame()

            while True:
                last_index = all_potters.rfind(' ', first_index, last_index)
                text, broken_text = self.extract_source_and_target_text(all_potters[first_index, last_index])
                df_line = pd.DataFrame()
                df_line['source_text'] = broken_text
                df_line['target_text'] = text
                df.append(df_line, ignore_index=True)

                first_index = last_index + 1
                last_index += 500

                if last_index > len(all_potters):
                    break

            if self.git_repository:
                pandas_df.to_csv(dataset_file_name, sep=';', index=False)

        pandas_df['target_text'] = pandas_df.apply(self.do_text_manipulation, axis=1)
        train, test = train_test_split(pandas_df, test_size=0.2)
        self.train(train_df=train, eval_df=test, outputdir=self.trained_model_dir)


def main():
    punctation = GermanPunctation()
    # ''Ein weiterer lauter Knall ertönte, und Dobby, Luna, Dean und Ollivander verschwanden.''
    print(punctation.punctate_text(f'ein weiterer lauter knall ertönte und dobby luna dean und ollivander verschwanden'))
    print(punctation.punctate_text(f'correct german grammar: ein weiterer lauter knall ertönte und dobby luna dean und ollivander verschwanden'))
    print(punctation.punctate_text(f'es ist schön so viele tolle menschen um sich zu haben denn ohne sie wäre es nicht so schön'))
    print(punctation.punctate_text(f'correct german grammar: es ist schön so viele tolle menschen um sich zu haben denn ohne sie wäre es nicht so schön'))


if __name__ == '__main__':
    main()
