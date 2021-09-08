import cherrypy
import numpy as np
from dictator.AudioTranslator import AudioTranslator
from GermanSpeechToTextTranslaterBase import GermanSpeechToTextTranslaterBase


class SrcTranslationServer:
    def __init__(self, translator: AudioTranslator):
        self.translator = translator

    @cherrypy.expose
    @cherrypy.tools.json_in()
    @cherrypy.tools.json_out()
    def translate_numpy_audio(self):
        data = cherrypy.request.json
        samples = np.array(data['audio'])
        output = self.translator.translate_numpy_audio(samples)
        print(f'Translated: {output}')
        return output


if __name__ == '__main__':
    config = {
        'global': {
            'server.socket_host': '127.0.0.1',
            'server.socket_port': 8080
        }
    }
    cherrypy.config.update(config)

    model_name = 'c:/share/NLP-Models/GermanWave2Vec/trained_model'
    translator = GermanSpeechToTextTranslaterBase(model_name=model_name, device='cpu')
    cherrypy.quickstart(SrcTranslationServer(translator))
