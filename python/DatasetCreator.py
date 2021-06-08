import os
import glob
import eyed3
import pandas as pd
from pathlib import Path
from Mp3VoiceSplitter import get_snippet_info_from_mp3_file


def create_content_in_snippet_directory(snippet_directory, translator=None, orig_from_title_tag=False):
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

        result.to_csv(f'{snippet_directory}/content.csv', sep=';', index=False)

        if translator:
            snippet_df['Translated0'] = translator.translate_audio(mp3_file_path)
            result.to_csv(f'{snippet_directory}/content-translated.csv', sep=';', index=False)

        if orig_from_title_tag:
            audiofile = eyed3.load(f'{abu_dir}/al0005.mp3')
            snippet_df['OriginalText'] = audiofile.tag.title

            if translator:
                result.to_csv(f'{snippet_directory}/content-translated-with_original.csv', sep=';', index=False)
            else:
                result.to_csv(f'{snippet_directory}/content-with_original.csv', sep=';', index=False)

        result = result.append(snippet_df, ignore_index = True)

    return result

