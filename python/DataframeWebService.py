import cherrypy
import pandas as pd


class GermanSpeechToTextTranslaterWebService:
   def __init__(self, translator):
       self.translator = translator

   @cherrypy.expose
   @cherrypy.tools.json_out()
   @cherrypy.tools.json_in()
   def translate_to_text(self):
      data = cherrypy.request.json
      df = pd.DataFrame(data)
      samples = json_to_audio(df)
      output = self.translator.translate_audio_samples(samples)
      return output.to_json()
	  
	  
if __name__ == '__main__':
   config = {'server.socket_host': '0.0.0.0'}
   cherrypy.config.update(config)
   cherrypy.quickstart(DataframeWebService())