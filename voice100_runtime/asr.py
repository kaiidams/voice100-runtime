# Copyright (C) 2021 Katsuya Iida. All rights reserved.

from typing import Text
import numpy as np
import onnxruntime as ort

from .text import (
    CharTokenizer,
    CMUTokenizer)
from .audio import MelSpectrogram


class ASR:
    def __init__(
        self, model_path: str,
        model_type: Text = None
    ) -> None:
        self.sample_rate = 16000
        self._model_type = model_type
        if self._model_type == "phone" or self._model_type == "phone_v2":
            self._tokenizer = CMUTokenizer()
        else:
            self._tokenizer = CharTokenizer()
        self._transform = MelSpectrogram()
        self._asr_sess = ort.InferenceSession(model_path)

    def __call__(
        self, waveform: np.ndarray, sample_rate: int, aligned: bool = True
    ) -> np.ndarray:
        if self._model_type == "v2" or self._model_type == "phone_v2":
            return self._v2(waveform, sample_rate, aligned)
        else:
            return self._v1(waveform, sample_rate, aligned)

    def _v1(
        self, waveform: np.ndarray, sample_rate: int, aligned: bool = True
    ) -> np.ndarray:
        assert sample_rate == self.sample_rate
        audio = self._transform(waveform)
        (logits,) = self._asr_sess.run(
            output_names=["logits"], input_feed={"audio": audio[np.newaxis, :, :]}
        )

        pred = np.argmax(logits, axis=2)
        aligntext = self._tokenizer.decode(pred[0])
        if not aligned:
            return aligntext
        text = self._tokenizer.merge_repeated(aligntext)
        return text

    def _v2(
        self, waveform: np.ndarray, sample_rate: int, aligned: bool = True
    ) -> np.ndarray:
        assert sample_rate == self.sample_rate
        audio = self._transform(waveform)
        audio_len = np.array(audio.shape[0], dtype=np.int64)
        (logits, _) = self._asr_sess.run(
            output_names=["logits", "logits_len"],
            input_feed={
                "audio": audio[np.newaxis, :, :],
                "audio_len": audio_len[np.newaxis]
            }
        )

        pred = np.argmax(logits, axis=2)
        aligntext = self._tokenizer.decode(pred[:, 0])
        if not aligned:
            return aligntext
        text = self._tokenizer.merge_repeated(aligntext)
        return text
