# Copyright (C) 2021 Katsuya Iida. All rights reserved.

from typing import Text, Union, Tuple
import numpy as np
import onnxruntime as ort

from .text import (
    make_aligntext,
    BasicPhonemizer, BasicTokenizer, CharTokenizer,
    CMUPhonemizer)
from .vocoder import WORLDVocoder


class TTS:
    def __init__(
        self, align_model_path: Text, audio_model_path: Text,
        model_type: Text = None
    ):
        self.sample_rate = 16000
        self._model_type = model_type
        if self._model_type == "mt":
            self._phonemizer = BasicPhonemizer()
            self._tokenizer = CharTokenizer()
            self._output_tokenizer = BasicTokenizer(language="en")
        elif self._model_type in "phone" or self._model_type == "phone_v2":
            self._phonemizer = CMUPhonemizer()
            self._tokenizer = BasicTokenizer(language="en")
        elif self._model_type in "ja_phone_v2":
            self._phonemizer = None
            self._tokenizer = BasicTokenizer(language="ja")
        else:
            self._phonemizer = BasicPhonemizer()
            self._tokenizer = CharTokenizer()
        self._vocoder = WORLDVocoder()
        self._align_sess = ort.InferenceSession(align_model_path)
        self._audio_sess = ort.InferenceSession(audio_model_path)

    def __call__(
        self, input_text: str, return_align: bool = False
    ) -> Union[Tuple[np.ndarray, int], Tuple[np.ndarray, int, Text]]:
        if self._model_type in ['v2', "phone_v2", "ja_phone_v2"]:
            return self._v2(input_text, return_align)
        else:
            return self._v1(input_text, return_align)

    def _v1(
        self, input_text: str, return_align: bool = False
    ) -> Union[Tuple[np.ndarray, int], Tuple[np.ndarray, int, Text]]:
        text = self._phonemizer(input_text)
        text = self._tokenizer(text)
        (align,) = self._align_sess.run(
            output_names=["align"], input_feed={"text": text[None, :]}
        )
        align = np.clip(align, a_min=0, a_max=None)
        align = np.exp(align) - 1
        align = align[0]
        aligntext = make_aligntext(text, align)
        if self._model_type == "mt":
            f0, logspc, codeap, logits = self._audio_sess.run(
                output_names=["f0", "logspc", "codeap", "logits"],
                input_feed={"aligntext": aligntext[None, :]},
            )
            waveform = self._vocoder.decode(f0[0], logspc[0], codeap[0])
            if return_align:
                outputtext = np.argmax(logits, axis=1)
                outputtext = self._output_tokenizer.decode(outputtext[0])
                return waveform, self.sample_rate, outputtext.split("/")
            return waveform, self.sample_rate
        else:
            f0, logspc, codeap = self._audio_sess.run(
                output_names=["f0", "logspc", "codeap"],
                input_feed={"aligntext": aligntext[None, :]},
            )
            waveform = self._vocoder.decode(f0[0], logspc[0], codeap[0])
            if return_align:
                aligntext = self._tokenizer.decode(aligntext)
                return waveform, self.sample_rate, aligntext
            return waveform, self.sample_rate

    def _v2(
        self, input_text: str, return_align: bool = False
    ) -> Union[Tuple[np.ndarray, int], Tuple[np.ndarray, int, Text]]:
        if self._phonemizer is not None:
            text = self._phonemizer(input_text)
        else:
            text = input_text
        text = self._tokenizer(text)
        text_len = np.array([text.shape[0]], dtype=np.int64)
        (align,) = self._align_sess.run(
            output_names=["align"],
            input_feed={"text": text[None, :], "text_len": text_len}
        )
        align = align[0]
        aligntext = make_aligntext(text, align)
        aligntext_len = np.array([aligntext.shape[0]], dtype=np.int64)
        f0, logspc, codeap = self._audio_sess.run(
            output_names=["f0", "logspc", "codeap"],
            input_feed={
                "aligntext": aligntext[None, :],
                "aligntext_len": aligntext_len
            },
        )
        waveform = self._vocoder.decode(f0[0], logspc[0], codeap[0])
        if return_align:
            aligntext = self._tokenizer.decode(aligntext)
            return waveform, self.sample_rate, aligntext
        return waveform, self.sample_rate
