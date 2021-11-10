# Copyright (C) 2021 Katsuya Iida. All rights reserved.

import re
import numpy as np

__all__ = [
    "BasicPhonemizer",
    "CharTokenizer",
]

DEFAULT_CHARACTERS = "_ abcdefghijklmnopqrstuvwxyz'"
NOT_DEFAULT_CHARACTERS_RX = re.compile("[^" + DEFAULT_CHARACTERS[1:] + "]")
DEFAULT_VOCAB_SIZE = len(DEFAULT_CHARACTERS)
assert DEFAULT_VOCAB_SIZE == 29


def make_aligntext(text, align, head=5, tail=5):
    aligntext_len = head + int(np.sum(align)) + tail
    aligntext = np.zeros(aligntext_len, dtype=text.dtype)
    t = head
    for i in range(align.shape[0]):
        t += align[i, 0].item()
        s = round(t)
        t += align[i, 1].item()
        e = round(t)
        if s == e:
            s = max(0, s - 1)
        for j in range(s, e):
            aligntext[j] = text[i]
    return aligntext


class BasicPhonemizer:
    def __init__(self):
        super().__init__()

    def __call__(self, text: str) -> str:
        return NOT_DEFAULT_CHARACTERS_RX.sub("", text.lower())


class CharTokenizer:
    def __init__(self, vocab=None):
        super().__init__()
        if vocab is None:
            vocab = DEFAULT_CHARACTERS
        self.vocab_size = len(vocab)
        self._vocab = vocab
        self._v2i = {x: i for i, x in enumerate(vocab)}

    def __call__(self, text: str) -> np.ndarray:
        return self.encode(text)

    def encode(self, text):
        encoded = [self._v2i[ch] for ch in text if ch in self._v2i]
        return np.array(encoded, dtype=np.int64)

    def decode(self, encoded) -> str:
        return "".join([self._vocab[x] for x in encoded if 0 <= x < len(self._vocab)])

    def merge_repeated(self, text: str) -> str:
        text = re.sub(r"(.)\1+", r"\1", text)
        text = text.replace("_", "")
        if text == " ":
            text = ""
        return text
