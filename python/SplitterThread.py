import pandas as pd
from threading import Thread, Event
from queue import Queue
from Mp3VoiceSplitter import join_frames


class SplitterThread(Thread):
    def __init__(
        self,
        snippet_path: str,
        audio_in_queue: Queue,
        vad_level=1,
        translator=None,
        translated_text_queue: Queue = None
    ):
        super().__init__(name='Splitter Thread')
        self.snippet_path = snippet_path
        self.translator = translator
        self.vad_level = vad_level
        self.audio_in_queue = audio_in_queue
        self.translated_text_queue = translated_text_queue

    def set_dataframe(self, df: pd.DataFrame):
        self.df = df

    def run(self):
        orig_mp3_file_name_without_extension = self.df['Orginaldatei'].values[0][0:-4]
        self.result = join_frames(
            destination_path=self.snippet_path,
            orig_mp3_file_name_without_extension=orig_mp3_file_name_without_extension,
            sample_rate=16_000,
            frame_queue=self.audio_in_queue,
            df=self.df,
            translator=self.translator,
            translated_text_queue=self.translated_text_queue,
            vad_level=self.vad_level
        )
