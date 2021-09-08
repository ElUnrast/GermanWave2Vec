import requests
from dictator.SrcAudioTools import load_mp3_as_sr16000


def main():
    samples, _ = load_mp3_as_sr16000('//matlab3/d/NLP-Data/audio/HspSections/HspSections-065/00007998.mp3')
    data = {'audio': samples.tolist()}

    url = 'http://127.0.0.1:8080/translate_numpy_audio'
    response = requests.post(url, json=data)
    print(response)
    print(f'Translation: {response.json()}')


if __name__ == '__main__':
    main()
