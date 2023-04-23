# Copyright (C) 2021 Katsuya Iida. All rights reserved.

from typing import Text, List
import os
import sys
import hashlib

CACHE_DIR = os.path.expanduser("~/.cache/voice100_runtime")

MODEL_URLS = {
    "asr_en": [
        "https://github.com/kaiidams/voice100-runtime/releases/download/v1.1.1/asr_en_conv_base_ctc-20220126.onnx",
        "92801e1e4927f345522706a553e86eebd1e347651620fc6d69bfa30ab4104b86",
        []
    ],
    "asr_en_phone": [
        "https://github.com/kaiidams/voice100-runtime/releases/download/v1.1.0/asr_en_phone_conv_base_ctc-20220115.onnx",
        "dff282a17ef8544f5f370becfab20f05bab4ada96f3b795fb28f7daec8e8efb5",
        []
    ],
    "asr_ja": [
        "https://github.com/kaiidams/voice100-runtime/releases/download/v0.2/stt_ja_conv_base_ctc-20211127.onnx",
        "adcde1c63a2fc23cfbad741fbbd69145353d21edf49f14695a56de33bc3a822e",
        []
    ],
    "tts_en": [
        "https://github.com/kaiidams/voice100-runtime/releases/download/v1.3.0/ttsalign_en_conv_base-20220409.onnx",
        "ef4bbd19e44052d16cc78b4a8fce6dc7783b407a66812bfdb97f042bf2e61817",
        "https://github.com/kaiidams/voice100-runtime/releases/download/v1.0.1/ttsaudio_en_conv_base-20220107.onnx",
        "a20fec366d1a4856006bbf7cfac7d989ef02b0c1af676c0b5e6f318751325a2f",
        []
    ],
    "tts_en_phone": [
        "https://github.com/kaiidams/voice100-runtime/releases/download/v1.3.0/ttsalign_en_phone_conv_base-20220409.onnx",
        "d3742bb01c5bf6840caa0b370e659f5a0fcaf6efb96b25a40c59c898b1abe28d",
        "https://github.com/kaiidams/voice100-runtime/releases/download/v1.1.0/ttsaudio_en_phone_conv_base-20220105.onnx",
        "f2240f7364497c4232772eee0737d38ad05f9d01c8d64cdb32f66ed93009095d",
        ["phone"]
    ],
    "tts_en_mt": [
        "https://github.com/kaiidams/voice100-runtime/releases/download/v1.3.0/ttsalign_en_conv_base-20220409.onnx",
        "ef4bbd19e44052d16cc78b4a8fce6dc7783b407a66812bfdb97f042bf2e61817",
        "https://github.com/kaiidams/voice100-runtime/releases/download/v1.2.0/ttsaudio_en_mt_conv_base-20220316.onnx",
        "5d0f426509bb662deab3ca9cf964f68dbaf2a30b55e653205c98eaad63978468",
        ["mt"]
    ],
    "tts_ja": [
        "https://github.com/kaiidams/voice100-runtime/releases/download/v1.3.0/ttsalign_ja_conv_base-20220411.onnx",
        "fcb5d8a457531ae8020c55d9e5bcdb814e27054fd78427837e6454635c041600",
        "https://github.com/kaiidams/voice100-runtime/releases/download/v1.3.1/ttsaudio_ja_conv_base-20220416.onnx",
        "af0add3d8f39316a126c4dc78b85a22b4e3dbe99853ef85e89c10ea7fdb0e6b3",
        []
    ],
    "asr_en_v2": [
        "https://github.com/kaiidams/voice100-runtime/releases/download/v1.4.0/asr_en_base-20230319.onnx",
        "6a284dbcdf88091faac962f6741b434d4c93c0d5a7f8085ad85198247fad25bc",
        ["v2"]
    ],
    "asr_en_phone_v2": [
        "https://github.com/kaiidams/voice100-runtime/releases/download/v1.4.0/asr_en_phone_base-20230314.onnx",
        "4db3799fb0e29bad2c1db32489679584c13f788e9d9f9f877819badc6875ac8a",
        ["phone", "v2"]
    ],
    "asr_ja_phone_v2": [
        "https://github.com/kaiidams/voice100-runtime/releases/download/v1.4.0/asr_ja_phone_base-20230104.onnx",
        "7418762b334705e8bed2c34ad5eab252e40502aa91a91c9fa0f49af761e24800",
        ["ja", "phone", "v2"]
    ],
    "tts_en_v2": [
        "https://github.com/kaiidams/voice100-runtime/releases/download/v1.4.0/align_en_base-20230401.onnx",
        "bfe28201ebebf5476518f3283b0471682d5f7f0e486fee288edd70219ba21e78",
        "https://github.com/kaiidams/voice100-runtime/releases/download/v1.4.0/tts_en_base-20230407.onnx",
        "0db072e76bc54a91a277b7d301083a59ae32cdec5add77aeb47cb192ce2b244d",
        ["v2"]
    ],
    "tts_en_phone_v2": [
        "https://github.com/kaiidams/voice100-runtime/releases/download/v1.4.0/align_en_phone_base-20230407.onnx",
        "eedf88e7dbe163faafc619c3aa3e645ec1833fa49dbb89a701c96ad4f71a57d0",
        "https://github.com/kaiidams/voice100-runtime/releases/download/v1.4.0/tts_en_phone_base-20230401.onnx",
        "75da86c9eed447889b4a13f777b25137d3208737b7de8c1130f4f4728a5909c6",
        ["phone", "v2"]
    ],
    "tts_ja_phone_v2": [
        "https://github.com/kaiidams/voice100-runtime/releases/download/v1.4.0/align_ja_phone_base-20230203.onnx",
        "9fdb9dd0bc8b6d13b0a22d20073b11f2ab11baaf7498a7cd098d37590712ceac",
        "https://github.com/kaiidams/voice100-runtime/releases/download/v1.4.0/tts_ja_phone_base-20230204.onnx",
        "dcdd08fe42660b49c4022fecfb867ad7bebc83d56ffaed68ea63595e615d73ee",
        ["ja", "phone", "v2"]
    ],
}


def _verify_hash(file: Text, hash: Text) -> None:
    m = hashlib.sha256()
    with open(file, 'rb') as fp:
        while True:
            x = fp.read(4096)
            if len(x) == 0:
                break
            m.update(x)
    if m.hexdigest() != hash:
        raise ValueError(f'The file "{file}" doesn\'t have correct SHA256 hash.')


def download_model(url: Text, hash: Text) -> Text:
    if not os.path.exists(CACHE_DIR):
        os.makedirs(CACHE_DIR)
    cached_file = os.path.join(CACHE_DIR, os.path.basename(url))
    if os.path.exists(cached_file):
        return cached_file
    sys.stderr.write(f'Downloading: "{url}" to {cached_file}\n')
    import urllib.request
    urllib.request.urlretrieve(url, cached_file)
    _verify_hash(cached_file, hash)
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
        model_path = download_model(MODEL_URLS[name][0], MODEL_URLS[name][1])
        model_type = MODEL_URLS[name][2]
        from .asr import ASR
        return ASR(model_path, model_type=model_type)
    elif name.startswith("tts_") and name in MODEL_URLS:
        model_type = MODEL_URLS[name][4]
        align_model_path = download_model(MODEL_URLS[name][0], MODEL_URLS[name][1])
        audio_model_path = download_model(MODEL_URLS[name][2], MODEL_URLS[name][3])
        from .tts import TTS
        return TTS(align_model_path, audio_model_path, model_type=model_type)
    else:
        raise ValueError(f"Unknown model {name}")
