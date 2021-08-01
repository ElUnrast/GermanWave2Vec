import re
from SnippetDatasets import SnippetDatasets

ersetzungen = {
    # first person verb append e
    re.compile(r"(\b)ich (steh|seh|würd|geb|heb|komm|geh|erklär)(\b)"): r'\1ich \2e\3',
    re.compile(r"(\b)(wundert|der|die|hätten|und|mich|mussten|mal|will|erzählen|kapiert|hört|wollte|man|gefällt|können|kümmert|stimmt|zeige|versuche|läuft|steht|darf|wartet|versuchen|verstehe|aber|ihr|erkläre|bin|wie|glaube|tun|gebe|wird|liegt|schlage|nimm|werden|sieht|würdet|hat|kann|habt|geht|bin|soll|hat|reicht|verstehe|würdet|wüsstet|sein|wenn|gibt|mache|heißt|hätte|sollen|werde|wäre|tut|habe|haben|sage|sagen|wäre|macht|legt|wir|war|ich|du|er|sie|gab|geht|gib|sind|ging|seht|probier|wollt|kommt)'s(\b)"): r'\1\2 es\3',
    # remove ' at word ending
    re.compile(r"(\b)(stirn|fern|all|allein|lass)'(\b)"): r'\1\2\3',
    # insert e instead of '
    re.compile(r"(\b)(hab|gern|werd|wär|glaub|tu|würd|hätt)'(\b)"): r'\1\2e\3',
    # insert en instead of 'n
    re.compile(r"(\b)(hätt)'n(\b) "): r'\1\2en\3',
    re.compile(r"(\b)aus'm(\b) "): r'\1aus dem\2',
    # remove ' in ...'s
    re.compile(r"(\b)(auf|durch)'s(\b)"): r'\1\2\3',
    # replace es -> s at word ending
    re.compile(r"(nebelig|stab|mitglied|turm|gericht|meer|trank|kobold|zustand|feld|fluch|geschöpf|jahr|wirt|schiff|schritt|mann|traum|wunsch|blick|freund|tag|geist|hund|kampf|tag|fall|sohn|mond|land|dorf|krieg|ort)es(\b)"): r'\1s\2',
    re.compile(r"(pfad|gang|haus|blick|tisch|raum|arm|baum|wort|wind|grad|kind|buch|haar|tag|mund|raum|grab|fall)es(\b)"): r'\1s\2',
    # insert e
    re.compile(r"(\b)wacklig(\b)"): r'\1wackelig\2',
    re.compile(r"(\b)knubblig(\b)"): r'\1knubbelig\2',
    re.compile(r"(\b)innern(\b)"): r'\1inneren\2',
    re.compile(r"(\b)gruslig(\b)"): r'\1gruselig\2',
    # replay complete word
    re.compile(r"(\b)okay(\b)"): r'\1ok\2',
    re.compile(r"(\b)zirka(\b)"): r'\1ca\2',
    re.compile(r"(\b)und so weiter(\b)"): r'\1usw\2',
    re.compile(r"(\b)mister(\b)"): r'\1mr\2',
    re.compile(r"(\b)misses(\b)"): r'\1mrs\2',
    re.compile(r"(\b)ham(\b)"): r'\1haben\2',
    # remove space beteen words
    re.compile(r"(\b)ex (\w+)"): r'\1ex\2',
    # remoove double vocals
    re.compile(r"(\b)([Oo])o+h(\b)"): r'\1\2h\3',
    re.compile(r"(\b)([Jj])a[ah]+(\b)"): r'\1\2a\3',
    re.compile(r"(\b)([Nn]ei)i+n+(\b)"): r'\1\2n\3',
    # wörtliche rede ...
    re.compile(r"(\b)isses(\b)"): r'\1ist es\2',
    re.compile(r"soll'n"): r'sollen',
    re.compile(r"sollt'n"): r'sollten',
    re.compile(r"krieg'n"): r'kriegen',
    re.compile(r"(\b)ham's"): r'\1haben es',
    re.compile(r"(\b)ham(\b)"): r'\1haben\2',
    re.compile(r" 'n(\b)"): r' ein\1',
    re.compile(r" 'ne(\b)"): r' eine\1',
    re.compile(r" 'nen(\b)"): r' einen\1',
    re.compile(r" 'nem(\b)"): r' einem\1',
    # Fleur
    re.compile(r"(\b)'Ochzeit(\b)"): r'\1Hochzeit\2',
    re.compile(r"(\b)'Arry(\b)"): r'\1Harry\2',
    re.compile(r"(\b)'abe(\b)"): r'\1habe\2',
    re.compile(r"(\b)'aben(\b)"): r'\1haben\2',
    re.compile(r"(\b)'ässlisch(\b)"): r'\1hässlisch\2',
    re.compile(r"(\b)unge'euer(\b)"): r'\1ungeheuer\2',
    re.compile(r"(\b)(Nn)atürlisch(\b)"): r'\1\2atürlich\3',
    re.compile(r"(\b)(Ii)sch(\b)"): r'\1\2ch\3',
    re.compile(r"(\b)(Nn)ischt(\b)"): r'\1\2icht\3',
    # ähms
    re.compile(r", ähm,(\b)"): r'\1',
    re.compile(r"\bähm "): '',
    re.compile(r"\bahm "): '',
}


def substitute(text):
    result = text

    for regex, replacement in ersetzungen.items():
        try:
            result = regex.sub(replacement, result)
        except:
            print(f'text: {text}, pattern: {regex.pattern}')
            result = text

    return result


def main():
    dataset_loader = SnippetDatasets(False, '//matlab3/D/NLP-Data/audio', 'C:/gitviews/GermanWave2Vec')

    for ds_id in dataset_loader.local_datasets.keys():
        if ds_id.endswith('FvM'):
            print(f'')
            ds = dataset_loader.load_ds_content_translated_with_original(ds_id, prune=False)
            wer = dataset_loader.get_word_error_rate(ds_id)
            ds_epoche = wer['trained_epochs']
            ds['OriginalText'] = ds['OriginalText'].apply(substitute)

            # for idx in range(len(ds)):
            #     original_text = ds.iloc[idx]['OriginalText']

            #     for regex, replacement in ersetzungen.items():
            #         ds['OriginalText'].values[idx] = re.sub(regex, replacement, original_text)

            dataset_loader.save_content_translated_with_original(ds_id, ds, ds_epoche)


if __name__ == '__main__':
    main()
