# Copyright (C) 2021 Katsuya Iida. All rights reserved.

import numpy as np
import onnxruntime as ort

from .text import CharTokenizer
from .audio import MelSpectrogram


class STT:
    def __init__(self, model_path: str) -> None:
        self.sample_rate = 16000
        self._tokenizer = CharTokenizer()
        self._transform = MelSpectrogram()
        self._stt_sess = ort.InferenceSession(model_path)

    def __call__(
        self, waveform: np.ndarray, sample_rate: int, aligned: bool = True
    ) -> np.ndarray:
        assert sample_rate == self.sample_rate
        audio = self._transform(waveform)
        (logits,) = self._stt_sess.run(
            output_names=["logits"], input_feed={"audio": audio[np.newaxis, :, :]}
        )

        pred = np.argmax(logits, axis=2)
        aligntext = self._tokenizer.decode(pred[0])
        if not aligned:
            return aligntext
        text = self._tokenizer.merge_repeated(aligntext)
        return text
