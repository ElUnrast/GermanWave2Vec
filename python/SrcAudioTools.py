import pandas as pd
import numpy as np
import pydub
import librosa
import json
import contextlib
import wave


def audio_to_json(audio_np_array):
    return pd.Series(audio_np_array).to_json(orient='values')


def json_to_audio(json_str):
    return np.array(json.loads(json_str))


def mp3_to_np(mp3_file: str):
    audio_segment = pydub.AudioSegment.from_mp3(mp3_file)
    return pydub_to_np(audio_segment)


def pydub_to_np(audio: pydub.AudioSegment):
    """Converts pydub audio segment into float32 np array of shape [channels, duration_in_seconds*sample_rate],
    where each value is in range [-1.0, 1.0]. Returns tuple (audio_np_array, sample_rate)"""
    # -> (np.ndarray, int)
    # get_array_of_samples returns the data in format:
    # [sample_1_channel_1, sample_1_channel_2, sample_2_channel_1, sample_2_channel_2, ....]
    # where samples are integers of sample_width bytes.
    return np.array(audio.get_array_of_samples(), dtype=np.float32).reshape(
        (-1, audio.channels)).T / (1 << (8 * audio.sample_width)), audio.frame_rate


def load_mp3_as_sr16000(audio_file_name: str):
    print(f'Loading: {audio_file_name}')

    if audio_file_name.endswith('.mp3'):
        # samples, sampling_rate = torchaudio.load(audio_file_name)
        samples, sampling_rate = mp3_to_np(audio_file_name)
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


def convert_numpy_samples_to_audio_bytes(samples, normalize=True):
    m = abs(samples).max() if normalize else 1
    s = samples * (32768 / m)
    s16 = s.astype(np.int16)
    il = s16.tolist()
    ba = bytearray()

    for i in il:
        ba.extend(i.to_bytes(2, 'little', signed=True))

    return bytes(ba)


def write_mp3(audio_file_name: str, audio: bytes, sample_rate=16_000):
    audio_segment = pydub.AudioSegment(
        audio,
        frame_rate=sample_rate,
        sample_width=2,
        channels=1
    )
    audio_segment.export(audio_file_name, format='mp3', codec='mp3')
    # ia1 = convert_audio_bytes_to_numpy(audio)
    # ia = np.array([ia1, ia1])
    # torchaudio.save(audio_file_name, torch.from_numpy(ia).float(), sample_rate, format='mp3')


def read_wave(path):
    """Reads a .wav file.
    Takes the path, and returns (PCM audio data, sample rate).
    """
    with contextlib.closing(wave.open(path, 'rb')) as wf:
        num_channels = wf.getnchannels()
        assert num_channels == 1
        sample_width = wf.getsampwidth()
        assert sample_width == 2
        sample_rate = wf.getframerate()
        assert sample_rate in (8000, 16000, 32000, 48000)
        pcm_data = wf.readframes(wf.getnframes())
        print(f'Read {path}, num_channels: {num_channels}, sample_width: {sample_width}, sample_rate: {sample_rate}')
        return pcm_data, sample_rate


def write_wave(path, audio, sample_rate):
    """Writes a .wav file.
    Takes path, PCM audio data, and sample rate.
    """
    print(f'Writing audio with type: {type(audio)} and len: {len(audio)}')
    with contextlib.closing(wave.open(path, 'wb')) as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(audio)
