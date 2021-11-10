# Copyright (C) 2021 Katsuya Iida. All rights reserved.

import numpy as np
import pyworld


class WORLDVocoder:
    def __init__(
        self,
        sample_rate: int = 16000,
        frame_period: int = 10.0,
        n_fft: int = None,
        log_offset: float = 1e-15,
    ) -> None:
        self.sample_rate = sample_rate
        self.frame_period = frame_period
        if n_fft is None:
            n_fft = 512
        self.n_fft = n_fft
        self.log_offset = log_offset

    def decode(
        self, f0: np.ndarray, logspc: np.ndarray, codeap: np.ndarray
    ) -> np.ndarray:
        f0 = f0.astype(np.double, order="C")
        logspc = logspc.astype(np.double, order="C")
        codeap = codeap.astype(np.double, order="C")
        spc = np.maximum(np.exp(logspc) - self.log_offset, 0).copy(order="C")
        ap = pyworld.decode_aperiodicity(codeap, self.sample_rate, self.n_fft)
        waveform = pyworld.synthesize(
            f0, spc, ap, self.sample_rate, frame_period=self.frame_period
        )
        return waveform
