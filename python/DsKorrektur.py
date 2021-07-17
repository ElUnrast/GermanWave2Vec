import re
from SnippetDatasets import SnippetDatasets


def main():
    ersetzungen = {
        re.compile(r"(\b)stirn' "): r'\1stirn ',
        re.compile(r"(\b)fern' "): r'\1fern ',
        re.compile(r"(\b)all' "): r'\1all ',
        re.compile(r"(\b)allein' "): r'\1allein ',
        re.compile(r"(\b)lass' "): r'\1lass ',

        re.compile(r"(\b)hab' "): r'\1habe ',
        re.compile(r"(\b)gern' "): r'\1gerne ',
        re.compile(r"(\b)werd' "): r'\1werde ',
        re.compile(r"(\b)wär' "): r'\1wäre ',
        re.compile(r"(\b)glaub' "): r'\1glaube ',
        re.compile(r"(\b)tu' "): r'\1tue ',
        re.compile(r"(\b)würd' "): r'\1würde ',
        re.compile(r"(\b)hätt'n "): r'\1hätten ',

        re.compile(r"(\b)auf's "): r'\1aufs ',
        re.compile(r"(\b)durch's "): r'\1durchs ',

        re.compile(r"(\b)ham(\b)"): r'\1haben\2',

        re.compile(r"(nebelig|stab|mitglied|turm|gericht|meer|trank|kobold|zustand|feld|fluch|geschöpf|jahr)es(\b)"): r'\1s\2',
        re.compile(r"(\b)wacklig(\b)"): r'\1wackelig\2',
        re.compile(r"(\b)knubblig(\b)"): r'\1knubbelig\2',
        re.compile(r"(\b)innern(\b)"): r'\1inneren\2',
        re.compile(r"(\b)gruslig(\b)"): r'\1gruselig\2',
        re.compile(r"(\b)okay(\b)"): r'\1ok\2',
        re.compile(r"(\b)ex (\w+)"): r'\1ex\2',
    }
    dataset_loader = SnippetDatasets(False, '//matlab3/D/NLP-Data/audio', 'C:/gitviews/GermanWave2Vec')

    for ds_id in dataset_loader.local_datasets.keys():
        if ds_id.endswith('FvM'):
            ds = dataset_loader.load_ds_content_translated_with_original(ds_id, prune=False)
            wer = dataset_loader.get_word_error_rate(ds_id)
            ds_epoche = wer['trained_epochs']

            for idx in range(len(ds)):
                original_text = ds.iloc[idx]['OriginalText']

                for regex, replacement in ersetzungen.items():
                    ds['OriginalText'].values[idx] = re.sub(regex, replacement, original_text)

            dataset_loader.save_content_translated_with_original(ds_id, ds, ds_epoche)


if __name__ == '__main__':
    main()
