import glob
import collections
import sys
import platform
import pandas as pd
from pathlib import Path
from queue import Queue
from typing import Iterator
from GermanSpeechToTextTranslaterBase import GermanSpeechToTextTranslaterBase
from SrcAudioTools import convert_audio_bytes_to_numpy, write_mp3, write_wave, load_mp3_as_sr16000, convert_numpy_samples_to_audio_bytes

import webrtcvad


class Frame(object):
    """Represents a "frame" of audio data."""

    def __init__(self, bytes, offset, timestamp, duration):
        self.bytes = bytes
        self.offset = offset
        self.timestamp = timestamp
        self.duration = duration

    def get_samples(self):
        return convert_audio_bytes_to_numpy(self.bytes)


class VoicedSnippet(object):
    def __init__(self, voiced_frames):
        self.bytes = b''.join([f.bytes for f in voiced_frames])
        self.start = voiced_frames[0].offset
        self.end = voiced_frames[-1].offset + len(voiced_frames[-1].bytes)
        self.length = len(voiced_frames)
        self.duration = sum([f.duration for f in voiced_frames])

    def get_samples(self):
        return convert_audio_bytes_to_numpy(self.bytes)

    def write_audio(self, destination_path, file_name_without_extension):
        if 'Windows' == platform.system():
            return self.write_wave(destination_path, file_name_without_extension)

        return self.write_mp3(destination_path, file_name_without_extension)

    def write_mp3(self, destination_path, file_name_without_extension):
        file_name = f'{file_name_without_extension}.mp3'
        print(f'Writing {file_name}, duration: {self.duration:f}ms')
        write_mp3(f'{destination_path}/{file_name}', self.bytes)
        return file_name

    def write_wave(self, destination_path, file_name_without_extension):
        file_name = f'{file_name_without_extension}.wav'
        print(f'Writing {file_name}, duration: {self.duration:f}ms')
        write_wave(f'{destination_path}/{file_name}', self.bytes)
        return file_name


def frame_generator(frame_duration_ms, audio, sample_rate) -> Iterator[Frame]:
    """Generates audio frames from PCM audio data.
    Takes the desired frame duration in milliseconds, the PCM data, and
    the sample rate.
    Yields Frames of the requested duration.
    """
    n = int(sample_rate * (frame_duration_ms / 1000.0) * 2)
    print(f'Framegenerator with {n} bytes length')
    offset = 0
    timestamp = 0.0
    duration = (float(n) / sample_rate) / 2.0

    while offset + n < len(audio):
        yield Frame(audio[offset:offset + n], offset, timestamp, duration)
        timestamp += duration
        offset += n


def vad_collector(sample_rate, vad, frame_queue: Queue) -> Iterator[VoicedSnippet]:
    num_padding_frames = 6
    # We use a deque for our sliding window/ring buffer.
    ring_buffer = collections.deque(maxlen=num_padding_frames)
    # We have two states: TRIGGERED and NOTTRIGGERED. We start in the NOTTRIGGERED state.
    triggered = False

    voiced_frames = []

    while True:
        frame = frame_queue.get()

        if not frame:
            frame_queue.task_done()
            break

        is_speech = vad.is_speech(frame.bytes, sample_rate)
        frame_queue.task_done()

        # sys.stdout.write('1' if is_speech else '0')
        if not triggered:
            ring_buffer.append((frame, is_speech))
            num_voiced = len([f for f, speech in ring_buffer if speech])

            if num_voiced > 0.5 * ring_buffer.maxlen:
                triggered = True
                # sys.stdout.write('+(%s)' % (ring_buffer[0][0].timestamp,))
                # We want to yield all the audio we see from now until
                # we are NOTTRIGGERED, but we have to start with the
                # audio that's already in the ring buffer.
                for f, _ in ring_buffer:
                    voiced_frames.append(f)

                ring_buffer.clear()
        else:
            # We're in the TRIGGERED state, so collect the audio data
            # and add it to the ring buffer.
            voiced_frames.append(frame)
            ring_buffer.append((frame, is_speech))
            num_unvoiced = len([f for f, speech in ring_buffer if not speech])
            # If more than 90% of the frames in the ring buffer are
            # unvoiced, then enter NOTTRIGGERED and yield whatever
            # audio we've collected.
            # num_unvoiced > (0.8 * ring_buffer.maxlen)
            if num_unvoiced == ring_buffer.maxlen:
                triggered = False

                if len(voiced_frames) > 30:
                    sys.stdout.write('-(%s)' % (frame.timestamp + frame.duration))
                    yield VoicedSnippet(voiced_frames)

                ring_buffer.clear()
                voiced_frames = []

    if frame and triggered:
        sys.stdout.write('-(%s)' % (frame.timestamp + frame.duration))
        sys.stdout.write('\n')
        # If we have any leftover voiced audio when we run out of input, yield it.
        if voiced_frames:
            yield VoicedSnippet(voiced_frames)


def join_frames(
    destination_path,
    orig_file_name_without_extension,
    sample_rate,
    frame_queue,
    df=pd.DataFrame(),
    translator: GermanSpeechToTextTranslaterBase = None,
    translated_text_queue=None,
    vad_level=1  # (0 - 3)
) -> pd.DataFrame:
    vad = webrtcvad.Vad(vad_level)
    snippets = vad_collector(sample_rate, vad, frame_queue)

    result = pd.DataFrame()

    for snippet in snippets:
        snippet_df = df.copy()
        snippet_df['Start'] = [snippet.start // 2]
        snippet_df['End'] = [snippet.end // 2]
        snippet_df['Length'] = [snippet.length]
        snippet_df['Duration'] = [snippet.duration]

        if translator:
            translation, samples_size = translator.translate_numpy_audio(snippet.get_samples())

            if not translation:
                continue

            print(f'Translation: {translation}')

            if translated_text_queue:
                print(f'Put Translation into queue')
                translated_text_queue.put(translation)

            snippet_df['Size'] = [samples_size]
            snippet_df['Translated0'] = [translation]

        new_file_fame_without_extension = f'{orig_file_name_without_extension}-{snippet.start:09d}'
        new_file_name = snippet.write_audio(destination_path, new_file_fame_without_extension)
        snippet_df['Datei'] = [new_file_name]
        result = result.append(snippet_df, ignore_index=True)

    ordered_column_names = ['Datei', 'Start', 'End', 'Length', 'Duration']

    if translator:
        ordered_column_names.extend(['Size', 'Translated0'])

    result = result[ordered_column_names]
    return result


def generate_file_name_without_extnsion(self, orig_file_name_without_extension):
    return


def split_mp3(mp3_file_path, destination_path, df=pd.DataFrame(), translator=None) -> pd.DataFrame:
    samples, _ = load_mp3_as_sr16000(mp3_file_path)
    audio_bytes = convert_numpy_samples_to_audio_bytes(samples)
    sample_rate = 16_000
    frame_duration_ms = 30
    frames = frame_generator(frame_duration_ms, audio_bytes, sample_rate)
    frame_queue = Queue()

    for frame in frames:
        frame_queue.put(frame)

    frame_queue.put(None)
    return join_frames(mp3_file_path, destination_path, sample_rate, frame_queue, df, translator)


def split_mp3s(ds_id, translator, source_dir, destination_dir) -> pd.DataFrame:
    mp3FilenamesList = glob.glob(f'{source_dir}/*.mp3')
    result = pd.DataFrame()

    for mp3_file_path in mp3FilenamesList:
        df = pd.DataFrame({'DsId': [ds_id], 'Orginaldatei': Path(mp3_file_path).name})
        result = result.append(split_mp3(mp3_file_path, destination_dir, df, translator), ignore_index=True)

    result.to_csv(f'{destination_dir}/content-translated.csv', sep=';', index=False)
    return result


def get_snippet_info_from_mp3_file(mp3_file_path):
    '''returns: 'file_name, start, end, length, duration, size' of the MP3-Snippet'''
    file_name = Path(mp3_file_path).name
    sample_rate = 16_000
    samples, sample_length = load_mp3_as_sr16000(mp3_file_path)
    audio = convert_numpy_samples_to_audio_bytes(samples)
    frame_duration_ms = 30
    frames = frame_generator(frame_duration_ms, audio, sample_rate)
    snippet = VoicedSnippet(list(frames))
    return file_name, snippet.start, snippet.end, snippet.length, snippet.duration, sample_length
