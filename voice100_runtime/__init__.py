# Copyright (C) 2021 Katsuya Iida. All rights reserved.

from typing import Text, List
import os
import sys

CACHE_DIR = os.path.expanduser("~/.cache/voice100_runtime")

MODEL_URLS = {
    "asr_en": [
        "https://github.com/kaiidams/voice100-runtime/releases/download/v1.1.1/asr_en_conv_base_ctc-20220126.onnx",
        []
    ],
    "asr_en_phone": [
        "https://github.com/kaiidams/voice100-runtime/releases/download/v1.1.0/asr_en_phone_conv_base_ctc-20220115.onnx",
        []
    ],
    "asr_ja": [
        "https://github.com/kaiidams/voice100-runtime/releases/download/v0.2/stt_ja_conv_base_ctc-20211127.onnx",
        []
    ],
    "tts_en": [
        "https://github.com/kaiidams/voice100-runtime/releases/download/v1.3.0/ttsalign_en_conv_base-20220409.onnx",
        "https://github.com/kaiidams/voice100-runtime/releases/download/v1.0.1/ttsaudio_en_conv_base-20220107.onnx",
        []
    ],
    "tts_en_phone": [
        "https://github.com/kaiidams/voice100-runtime/releases/download/v1.3.0/ttsalign_en_phone_conv_base-20220409.onnx",
        "https://github.com/kaiidams/voice100-runtime/releases/download/v1.1.0/ttsaudio_en_phone_conv_base-20220105.onnx",
        ["phone"]
    ],
    "tts_en_mt": [
        "https://github.com/kaiidams/voice100-runtime/releases/download/v1.3.0/ttsalign_en_conv_base-20220409.onnx",
        "https://github.com/kaiidams/voice100-runtime/releases/download/v1.2.0/ttsaudio_en_mt_conv_base-20220316.onnx",
        ["mt"]
    ],
    "tts_ja": [
        "https://github.com/kaiidams/voice100-runtime/releases/download/v1.3.0/ttsalign_ja_conv_base-20220411.onnx",
        "https://github.com/kaiidams/voice100-runtime/releases/download/v1.3.1/ttsaudio_ja_conv_base-20220416.onnx",
        []
    ],
    "asr_en_v2": [
        "https://github.com/kaiidams/voice100-runtime/releases/download/v1.4.0/asr_en_base-20230319.onnx",
        ["v2"]
    ],
    "asr_en_phone_v2": [
        "https://github.com/kaiidams/voice100-runtime/releases/download/v1.4.0/asr_en_phone_base-20230314.onnx",
        ["phone", "v2"]
    ],
    "asr_ja_phone_v2": [
        "https://github.com/kaiidams/voice100-runtime/releases/download/v1.4.0/asr_ja_phone_base-20230104.onnx",
        ["ja", "phone", "v2"]
    ],
    "tts_en_v2": [
        "https://github.com/kaiidams/voice100-runtime/releases/download/v1.4.0/align_en_base-20230401.onnx",
        "https://github.com/kaiidams/voice100-runtime/releases/download/v1.4.0/tts_en_base-20230407.onnx",
        ["v2"]
    ],
    "tts_en_phone_v2": [
        "https://github.com/kaiidams/voice100-runtime/releases/download/v1.4.0/align_en_phone_base-20230407.onnx",
        "https://github.com/kaiidams/voice100-runtime/releases/download/v1.4.0/tts_en_phone_base-20230401.onnx",
        ["phone", "v2"]
    ],
    "tts_ja_phone_v2": [
        "https://github.com/kaiidams/voice100-runtime/releases/download/v1.4.0/align_ja_phone_base-20230203.onnx",
        "https://github.com/kaiidams/voice100-runtime/releases/download/v1.4.0/tts_ja_phone_base-20230204.onnx",
        ["ja", "phone", "v2"]
    ],
}


def download_model(url: Text) -> Text:
    if not os.path.exists(CACHE_DIR):
        os.makedirs(CACHE_DIR)
    cached_file = os.path.join(CACHE_DIR, os.path.basename(url))
    if os.path.exists(cached_file):
        return cached_file
    sys.stderr.write('Downloading: "{}" to {}\n'.format(url, cached_file))
    import urllib.request

    urllib.request.urlretrieve(url, cached_file)
    return cached_file


def list_models() -> List[Text]:
    """List names of all available models."""
    return list(MODEL_URLS.keys())


def load(name: Text):
    """Load a model"""
    # For compatibility
    if name == "stt_en":
        name = "asr_en"
    elif name == "stt_ja":
        name = "asr_ja"

    if name.startswith("asr_") and name in MODEL_URLS:
        model_path = download_model(MODEL_URLS[name][0])
        model_type = MODEL_URLS[name][1]
        from .asr import ASR
        return ASR(model_path, model_type=model_type)
    elif name.startswith("tts_") and name in MODEL_URLS:
        model_type = MODEL_URLS[name][2]
        align_model_path = download_model(MODEL_URLS[name][0])
        audio_model_path = download_model(MODEL_URLS[name][1])
        from .tts import TTS
        return TTS(align_model_path, audio_model_path, model_type=model_type)
    else:
        raise ValueError(f"Unknown model {name}")
