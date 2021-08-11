# Copyright (C) 2021 Katsuya Iida. All rights reserved.

def from_pretrained(name):
    if name == 'stt_en':
        from .stt import STT
        return STT()
    elif name == 'tts_en':
        from .tts import TTS
        return TTS()
    else:
        raise ValueError(f"Unknown model {name}")