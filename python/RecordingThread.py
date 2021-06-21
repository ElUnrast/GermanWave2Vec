import sounddevice as sd
import numpy as np
from queue import Queue
from threading import Thread, Event
from Mp3VoiceSplitter import Frame
from SrcAudioTools import convert_numpy_samples_to_audio_bytes


class RecordingThread(Thread):
    def __init__(self, audio_in_queue: Queue = None, input_device_index=None, output_device_index=None):
        super().__init__(name='Recording Thread')
        self.audio_in_queue = audio_in_queue
        self.input_device_index = 1  # input_device_index
        self.output_device_index = 4  # output_device_index

    def run(self):
        self.offset = 0
        self.timestamp = 0.0
        self.duration = 30.0
        self.blocksize = 480  # entspricht 30ms
        self.event = Event()
        with sd.InputStream(
            # device=(self.input_device_index, self.output_device_index),
            samplerate=16_000,
            blocksize=self.blocksize,
            dtype=np.float32,
            # latency=self.SOUND_DEVICE_LATENCY,
            channels=1,
            callback=self.callback
        ) as self.stream:
            self.event.wait()
            print('Finished')

    def terminate(self):
        # https://stackoverflow.com/questions/66964597/python-gui-freezing-problem-of-thread-using-tkinter-and-sounddevice
        self.stream.abort()  # abort the stream processing
        self.event.set()  # break self.event.wait()

        if self.audio_in_queue != None:
            self.audio_in_queue.put(None)

    def callback(self, indata, outdata, frames, time, status=None):
        """This is called (from a separate thread) for each audio block."""
        if self.audio_in_queue == None:
            print(f'data: {len(indata)}, status: {status}, frames: {frames}, time: {time}', flush=True)
        else:
            audio = convert_numpy_samples_to_audio_bytes(indata[:, 0], normalize=False)
            self.audio_in_queue.put(Frame(audio, self.offset, self.timestamp, self.duration))
            self.timestamp += self.duration
            self.offset += self.blocksize
