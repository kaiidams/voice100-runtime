# Copyright (C) 2021 Katsuya Iida. All rights reserved.

import numpy as np
import onnxruntime as ort
from typing import Tuple

from .text import make_aligntext, BasicPhonemizer, CharTokenizer
from .vocoder import WORLDVocoder


class TTS:
    def __init__(self, align_model_path, audio_model_path):
        self.sample_rate = 16000
        self._phonemizer = BasicPhonemizer()
        self._tokenizer = CharTokenizer()
        self._vocoder = WORLDVocoder()
        self._align_sess = ort.InferenceSession(align_model_path)
        self._audio_sess = ort.InferenceSession(audio_model_path)

    def __call__(self, input_text: str) -> Tuple[np.ndarray, int]:
        text = self._phonemizer(input_text)
        text = self._tokenizer(text)
        (align,) = self._align_sess.run(
            output_names=["align"], input_feed={"text": text[None, :]}
        )
        align = np.clip(align, a_min=0, a_max=None)
        align = np.exp(align) - 1
        align = align[0]
        aligntext = make_aligntext(text, align)
        f0, logspc, codeap = self._audio_sess.run(
            output_names=["f0", "logspc", "codeap"],
            input_feed={"aligntext": aligntext[None, :]},
        )
        waveform = self._vocoder.decode(f0[0], logspc[0], codeap[0])
        return waveform, self.sample_rate
