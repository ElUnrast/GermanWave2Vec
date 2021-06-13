import glob
import torch
import torchaudio
import librosa
import collections
import sys
import pandas as pd
import numpy as np
from pathlib import Path

import webrtcvad


def load_mp3_as_sr16000(audio_file_name: str):
    print(f'Loading: {audio_file_name}')

    if audio_file_name.endswith('.mp3'):
        samples, sampling_rate = torchaudio.load(audio_file_name)
        samples = np.asarray(samples[0])

        if sampling_rate != 16_000:
            samples = librosa.resample(samples, sampling_rate, 16_000)

        print(f'Audio Type: {type(samples)}, {type(samples[0])}')

        return samples, samples.shape[0]

    raise ValueError


def convert_audio_bytes_to_numpy(audio: bytes, normalize=True):
    il = []
    k = 0

    for i in range(0, len(audio), 2):
        b = audio[i:i+2]
        j = int.from_bytes(b, 'little', signed=True)
        il.append(j)
        k = k + 1

    result = np.array(il, dtype=np.float32)

    if normalize:
        max_peak = abs(result).max()

        if max_peak > 0:
            return result / max_peak
        else:
            return result
    else:
        return result / 32768


def convert_numpy_samples_to_audio_bytes(samples):
    m = abs(samples).max()
    s = samples * (32768 / m)
    s16 = s.astype(np.int16)
    il = s16.tolist()
    ba = bytearray()

    for i in il:
        ba.extend(i.to_bytes(2, 'little', signed=True))

    return bytes(ba)


def write_mp3(audio_file_name: str, audio: bytes, sample_rate=16_000):
    ia1 = convert_audio_bytes_to_numpy(audio)
    ia = np.array([ia1, ia1])
    torchaudio.save(audio_file_name, torch.from_numpy(ia).float(), sample_rate, format='mp3')


class Frame(object):
    """Represents a "frame" of audio data."""

    def __init__(self, bytes, offset, timestamp, duration):
        self.bytes = bytes
        self.offset = offset
        self.timestamp = timestamp
        self.duration = duration


class VoicedSnippet(object):
    def __init__(self, voiced_frames):
        self.bytes = b''.join([f.bytes for f in voiced_frames])
        self.start = voiced_frames[0].offset
        self.end = voiced_frames[-1].offset + len(voiced_frames[-1].bytes)
        self.length = len(voiced_frames)
        self.duration = sum([f.duration for f in voiced_frames])

    def get_samples(self):
        return convert_audio_bytes_to_numpy(self.bytes)

    def write_mp3(self, destination_path, orig_mp3_file_name_without_extension):
        new_file_name = f'{orig_mp3_file_name_without_extension}-{self.start:09d}.mp3'
        print(f'Writing {new_file_name}, duration: {self.duration:f}ms')
        write_mp3(f'{destination_path}/{new_file_name}', self.bytes)
        return new_file_name


def frame_generator(frame_duration_ms, audio, sample_rate):
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


def vad_collector_orig(sample_rate, frame_duration_ms,
                       padding_duration_ms, vad, frames):
    """Filters out non-voiced audio frames.
    Given a webrtcvad.Vad and a source of audio frames, yields only
    the voiced audio.
    Uses a padded, sliding window algorithm over the audio frames.
    When more than 90% of the frames in the window are voiced (as
    reported by the VAD), the collector triggers and begins yielding
    audio frames. Then the collector waits until 90% of the frames in
    the window are unvoiced to detrigger.
    The window is padded at the front and back to provide a small
    amount of silence or the beginnings/endings of speech around the
    voiced frames.
    Arguments:
    sample_rate - The audio sample rate, in Hz.
    frame_duration_ms - The frame duration in milliseconds.
    padding_duration_ms - The amount to pad the window, in milliseconds.
    vad - An instance of webrtcvad.Vad.
    frames - a source of audio frames (sequence or generator).
    Returns: A generator that yields PCM audio data.
    """
    num_padding_frames = int(padding_duration_ms / frame_duration_ms)
    # We use a deque for our sliding window/ring buffer.
    ring_buffer = collections.deque(maxlen=num_padding_frames)
    # We have two states: TRIGGERED and NOTTRIGGERED. We start in the
    # NOTTRIGGERED state.
    triggered = False

    voiced_frames = []
    for frame in frames:
        is_speech = vad.is_speech(frame.bytes, sample_rate)

        sys.stdout.write('1' if is_speech else '0')
        if not triggered:
            ring_buffer.append((frame, is_speech))
            num_voiced = len([f for f, speech in ring_buffer if speech])
            # If we're NOTTRIGGERED and more than 90% of the frames in
            # the ring buffer are voiced frames, then enter the
            # TRIGGERED state.
            if num_voiced > 0.9 * ring_buffer.maxlen:
                triggered = True
                sys.stdout.write('+(%s)' % (ring_buffer[0][0].timestamp,))
                # We want to yield all the audio we see from now until
                # we are NOTTRIGGERED, but we have to start with the
                # audio that's already in the ring buffer.
                for f, s in ring_buffer:
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
            if num_unvoiced > 0.9 * ring_buffer.maxlen:
                sys.stdout.write('-(%s)' % (frame.timestamp + frame.duration))
                triggered = False
                yield VoicedSnippet(voiced_frames)
                ring_buffer.clear()
                voiced_frames = []

    if triggered:
        sys.stdout.write('-(%s)' % (frame.timestamp + frame.duration))

    sys.stdout.write('\n')
    # If we have any leftover voiced audio when we run out of input,
    # yield it.
    if voiced_frames:
        yield VoicedSnippet(voiced_frames)


def vad_collector(sample_rate, vad, frames):
    num_padding_frames = 5
    # We use a deque for our sliding window/ring buffer.
    ring_buffer = collections.deque(maxlen=num_padding_frames)
    # We have two states: TRIGGERED and NOTTRIGGERED. We start in the
    # NOTTRIGGERED state.
    triggered = False

    voiced_frames = []
    for frame in frames:
        is_speech = vad.is_speech(frame.bytes, sample_rate)

        sys.stdout.write('1' if is_speech else '0')
        if not triggered:
            ring_buffer.append((frame, is_speech))
            num_voiced = len([f for f, speech in ring_buffer if speech])
            # If we're NOTTRIGGERED and more than 90% of the frames in
            # the ring buffer are voiced frames, then enter the
            # TRIGGERED state.
            if num_voiced > 0.8 * ring_buffer.maxlen:
                triggered = True
                sys.stdout.write('+(%s)' % (ring_buffer[0][0].timestamp,))
                # We want to yield all the audio we see from now until
                # we are NOTTRIGGERED, but we have to start with the
                # audio that's already in the ring buffer.
                for f, s in ring_buffer:
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
            if num_unvoiced > 0.8 * ring_buffer.maxlen:
                sys.stdout.write('-(%s)' % (frame.timestamp + frame.duration))
                triggered = False
                yield VoicedSnippet(voiced_frames)
                ring_buffer.clear()
                voiced_frames = []

    if triggered:
        sys.stdout.write('-(%s)' % (frame.timestamp + frame.duration))

    sys.stdout.write('\n')
    # If we have any leftover voiced audio when we run out of input,
    # yield it.
    if voiced_frames:
        yield VoicedSnippet(voiced_frames)


def split_mp3(mp3_file_path, destination_path, df=pd.DataFrame(), translator=None):
    sample_rate = 16_000
    samples, sample_length = load_mp3_as_sr16000(mp3_file_path)
    audio = convert_numpy_samples_to_audio_bytes(samples)
    vad = webrtcvad.Vad(1)
    frame_duration_ms = 30
    frames = frame_generator(frame_duration_ms, audio, sample_rate)
    frames = list(frames)
    # snippets = vad_collector_orig(sample_rate, frame_duration_ms, 300, vad, frames)
    snippets = vad_collector(sample_rate, vad, frames)

    orig_mp3_file_name = Path(mp3_file_path).name
    orig_mp3_file_name_without_extension = orig_mp3_file_name[0:-4]
    result = pd.DataFrame()

    for snippet in snippets:
        new_file_name = snippet.write_mp3(destination_path, orig_mp3_file_name_without_extension)
        snippet_df = df.copy()
        snippet_df['Datei'] = [new_file_name]
        snippet_df['Start'] = [snippet.start]
        snippet_df['End'] = [snippet.end]
        snippet_df['Length'] = [snippet.length]
        snippet_df['Duration'] = [snippet.duration]

        if translator:
            translation, samples_size = translator.translate_audio(f'{destination_path}/{new_file_name}')
            snippet_df['Size'] = [samples_size]
            snippet_df['Translated0'] = [translation]

        result = result.append(snippet_df, ignore_index=True)

    return result


def split_mp3s(ds_id, translator, source_dir, destination_dir):
    mp3FilenamesList = glob.glob(f'{source_dir}/*.mp3')
    result = pd.DataFrame()

    for mp3_file_path in mp3FilenamesList:
        df = pd.DataFrame({'DsId': [ds_id], 'Orginaldatei': Path(mp3_file_path).name})
        result = result.append(split_mp3(mp3_file_path, destination_dir, df, translator), ignore_index=True)

    result.to_csv(f'{destination_dir}/content-translated.csv', sep=';', index=False)
    return result


'''returns: 'file_name, start, end, length, duration, size' of the MP3-Snippet'''


def get_snippet_info_from_mp3_file(mp3_file_path):
    file_name = Path(mp3_file_path).name
    sample_rate = 16_000
    samples, sample_length = load_mp3_as_sr16000(mp3_file_path)
    audio = convert_numpy_samples_to_audio_bytes(samples)
    frame_duration_ms = 30
    frames = frame_generator(frame_duration_ms, audio, sample_rate)
    snippet = VoicedSnippet(list(frames))
    return file_name, snippet.start, snippet.end, snippet.length, snippet.duration, sample_length
