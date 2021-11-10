# Copyright (C) 2021 Katsuya Iida. All rights reserved.

import numpy as np
import librosa


class MelSpectrogram:
    def __init__(
        self,
        sample_rate=16000,
        n_fft=512,
        hop_length=160,
        win_length=400,
        n_mels=64,
        log_offset: float = 1e-6,
    ) -> None:
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.n_mels = n_mels
        self.log_offset = log_offset

    def __call__(self, waveform: np.ndarray) -> np.ndarray:
        audio: np.ndarray = librosa.feature.melspectrogram(
            waveform,
            sr=self.sample_rate,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            center=True,
            pad_mode="reflect",
            power=2.0,
            n_mels=self.n_mels,
            norm=None,
            htk=True,
        )
        audio = np.log(audio.T + self.log_offset)
        return audio.astype(np.float32)
