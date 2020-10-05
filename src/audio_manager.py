import typing
import simpleaudio as sa
import numpy as np

class SoundFactory:
    def create_audio(self, seconds: float, frequency: float):
        fs = 44100

        t = np.linspace(0, seconds, seconds * fs, False)
        note = np.sin(frequency * t * 2 * np.pi)

        audio = note * (2**15 - 1) / np.max(np.abs(note))
        audio = audio.astype(np.int16)

        return Sound(audio, fs)

class Sound:
    def __init__(self, audio_data_array: np.ndarray, frequency: float):
        self._sound_data = audio_data_array
        self.frequency = frequency

    def play(self):
        sa.play_buffer(self._sound_data, 1, 2, self.frequency)