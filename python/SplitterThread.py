import pandas as pd
from threading import Thread, Event
from queue import Queue
from Mp3VoiceSplitter import join_frames


class SplitterThread(Thread):
    def __init__(self, snippet_path: str, q: Queue, translator=None):
        super(SplitterThread, self).__init__(name='Splitter Thread')
        self.snippet_path = snippet_path
        self.translator = translator
        self.q = q

    def set_dataframe(self, df: pd.DataFrame):
        self.df = df

    def run(self):
        self.result = join_frames(
            self.df['Orginaldatei'].values[0],
            self.snippet_path,
            16_000,
            self.q,
            self.df,
            self.translator
        )
