import requests
import numpy as np
from SrcAudioTools import load_mp3_as_sr16000
from AudioTranslator import AudioTranslator


class SrcTranslationService(AudioTranslator):
    def __init__(self, server_address: str):
        self.url = f'http://{server_address}/translate_numpy_audio'
        print(f'Server Address: {self.url}')

    def translate_numpy_audio(self, samples: np.ndarray) -> str:
        print(f'Server Address: {self.url}')
        data = {'audio': samples.tolist()}
        response = requests.post(self.url, json=data)
        return (f'{response.json()}')


if __name__ == '__main__':
    translator = SrcTranslationService('127.0.0.1:8080')
    print(f'Server Address: {translator.self.url}')

    samples, _ = load_mp3_as_sr16000('//matlab3/d/NLP-Data/audio/HspSections/HspSections-065/00007998.mp3')
    response = translator.translate_numpy_audio(samples=samples)
    print(f'Translation: {response}')
