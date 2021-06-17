import os
import re
import glob
import eyed3
import pandas as pd
from Mp3VoiceSplitter import get_snippet_info_from_mp3_file

_regex_spaces = re.compile(r"  +")
_regex_number = re.compile(r"\d+")


def create_content_in_snippet_directory(snippet_directory, translator, orig_from_title_tag=False):
    ds_id = os.path.basename(snippet_directory)
    mp3FilenamesList = glob.glob(f'{snippet_directory}/*.mp3')
    # Autor;Sprecher;Titel;Orginaldatei;Datei;Start;End;Length
    result = pd.DataFrame()

    for mp3_file_path in mp3FilenamesList:
        print(f'- {mp3_file_path}')
        file_name, start, end, length, duration, sample_length = get_snippet_info_from_mp3_file(mp3_file_path)
        snippet_df = pd.DataFrame({
            'DsId': [ds_id],
            'Orginaldatei': [file_name],
            'Datei': [file_name],
            'Start': [start],
            'End': [end],
            'Length': [length],
            'Duration': [duration],
            'Size': [sample_length]
        })

        translation, _ = translator.translate_audio(mp3_file_path)
        snippet_df['Translated0'] = translation

        if not orig_from_title_tag:
            result = result.append(snippet_df, ignore_index=True)
            result.to_csv(f'{snippet_directory}/content-translated.csv', sep=';', index=False)
        else:
            orig_text, action = _get_text_from_title_tag(mp3_file_path)
            snippet_df['OriginalText'] = orig_text
            snippet_df['Action'] = action
            result = result.append(snippet_df, ignore_index=True)
            result.to_csv(f'{snippet_directory}/content-translated-with_original.csv', sep=';', index=False)

    return result


def _get_text_from_title_tag(mp3_file_path):
    audiofile = eyed3.load(mp3_file_path)
    orig_text = audiofile.tag.title.lower()

    if '[' in orig_text:
        return orig_text, 'exclude6'
    elif _regex_number.search(orig_text):
        return orig_text, 'exclude7'

    new_orig_text = ''

    for c in orig_text:
        if c in "abcdefghijklmnopqrstuvwxyzäöüß' ":
            new_orig_text += c

    return _regex_spaces.sub(' ', new_orig_text), 'train'
