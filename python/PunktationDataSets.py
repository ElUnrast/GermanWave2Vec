import re
import phonetics
import random
import glob
import pandas as pd


class PunktationDataSets:
    def __init__(self, max_token_length=196):
        self.max_token_length = max_token_length
        self.chars_to_ignore_regex = re.compile(r"[^A-Za-z0-9'öäüÖÄÜß,;.:?!»«… -']+")
        self.broken_chars_to_ignore_regex = re.compile(r"[^A-Za-z0-9'öäüÖÄÜß -]+")
        self.duplicate_space_regex = re.compile(r'  +')

    def extract_source_and_target_text(self, text):
        target_text = self.duplicate_space_regex.sub(' ', self.chars_to_ignore_regex.sub('', text))
        source_text = self.duplicate_space_regex.sub(' ', self.broken_chars_to_ignore_regex.sub('', target_text.lower()))
        return source_text, target_text

    def prepare_punctation_dataset(self, txt_directory):
        source_text_list = []
        target_text_list = []
        ende_zeichen = ',;.:?!«'

        for text_file in glob.glob(f'{txt_directory}/*.txt'):
            all_text = ''
            first = True

            with open(text_file, encoding='utf8') as f:
                print(f'reading: {text_file}')
                for line in f:
                    if not first:
                        all_text += ' '
                    first = False
                    all_text += line.strip()

            first_index = 0
            last_index = self.max_token_length

            while True:
                print(f'first: {first_index}, last: {last_index}')
                idx = -1

                for z in ende_zeichen:
                    idx = max(idx, all_text.rfind(z, first_index, last_index))

                if idx > first_index:
                    last_index = idx + 1
                else:
                    idx = all_text.rfind(' ', first_index, last_index)

                if last_index < 0:
                    raise ValueError()

                source_text, target_text = self.extract_source_and_target_text(all_text[first_index:last_index])
                source_text_list.append(source_text)
                target_text_list.append(target_text)

                first_index = last_index + 1
                last_index += self.max_token_length

                if last_index > len(all_text):
                    break

        pandas_df = pd.DataFrame()
        pandas_df['source_text'] = source_text_list
        pandas_df['target_text'] = target_text_list
        # pandas_df.to_csv(dataset_file_name, sep=';', index=False)
        return pandas_df


def main():
    punctation = PunktationDataSets()
    git_repository = 'C:/gitviews/GermanWave2Vec'
    txt_directory = f'{git_repository}/datasets/punkctation'
    potter_df = punctation.prepare_punctation_dataset(txt_directory)
    dataset_file_name = f'{git_repository}/datasets/punkctation/potter_punktation.csv'
    potter_df.to_csv(dataset_file_name, sep=';', index=False)


if __name__ == '__main__':
    main()
