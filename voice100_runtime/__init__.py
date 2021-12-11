# Copyright (C) 2021 Katsuya Iida. All rights reserved.

import os
import sys

CACHE_DIR = os.path.expanduser("~/.cache/voice100_runtime")

MODEL_URLS = {
    "stt_en": [
        "https://github.com/kaiidams/voice100-runtime/releases/download/v0.2/stt_en_conv_base_ctc-20211125.onnx"
    ],
    "tts_en": [
        "https://github.com/kaiidams/voice100-runtime/releases/download/v0.1/ttsalign_en_conv_base-20210808.onnx",
        "https://github.com/kaiidams/voice100-runtime/releases/download/v0.1/ttsaudio_en_conv_base-20210811.onnx"
    ],
    "stt_ja": [
        "https://github.com/kaiidams/voice100-runtime/releases/download/v0.2/stt_ja_conv_base_ctc-20211127.onnx"
    ],
    "tts_ja": [
        "https://github.com/kaiidams/voice100-runtime/releases/download/v0.2/ttsalign_ja_conv_base-20211118.onnx",
        "https://github.com/kaiidams/voice100-runtime/releases/download/v0.2/ttsaudio_ja_conv_base-20211118.onnx"
    ],
}


def download_model(url: str) -> str:
    if not os.path.exists(CACHE_DIR):
        os.makedirs(CACHE_DIR)
    cached_file = os.path.join(CACHE_DIR, os.path.basename(url))
    if os.path.exists(cached_file):
        return cached_file
    sys.stderr.write('Downloading: "{}" to {}\n'.format(url, cached_file))
    import urllib.request

    urllib.request.urlretrieve(url, cached_file)
    return cached_file


def load(name):
    if name == "stt_en" or name == "stt_ja":
        model_path = download_model(MODEL_URLS[name][0])
        from .stt import STT

        return STT(model_path)
    elif name == "tts_en" or name == "tts_ja":
        align_model_path = download_model(MODEL_URLS[name][0])
        audio_model_path = download_model(MODEL_URLS[name][1])
        from .tts import TTS

        return TTS(align_model_path, audio_model_path)
    else:
        raise ValueError(f"Unknown model {name}")
