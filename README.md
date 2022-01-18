# Voice100 Runtime

Voice100 Android App is a TTS/ASR sample app that uses
[ONNX Runtime](https://github.com/microsoft/onnxruntime/),
[WORLD](https://github.com/mmorise/World)
and [Voice100](https://github.com/kaiidams/voice100) neural TTS/ASR models
on Python.
Inference of Voice100 is low cost as its models are tiny and only depend
on CNN without recursion.

- Beginnings are apt to be determinative and when reinforced 
by continuous applications of similar influence. [Sample audio](sample.wav)
- mata toojinoyoonigodaimyooootoyobarerushuyoonamyoooonochuuoonihaisarerukotomoooi
[Japanese sample audio](sample_ja.wav)

## Install

Use `pip` to install from GitHub.

```sh
pip install git+https://github.com/kaiidams/voice100-runtime.git
```

## List available models

```python
import voice100_runtime
print(voice100_runtime.list_models())
```

## Using TTS

This downloads ONNX files in `~/.cache/voice100_runtime/` if not
available.

```python
import soundfile as sf
import voice100_runtime
tts = voice100_runtime.load("tts_en")
waveform, sample_rate = tts("Hello, world!")
sf.write("output.wav", waveform, sample_rate, "PCM_16")
```

## Using ASR

This downloads an ONNX file in `~/.cache/voice100_runtime/` if not
available.

```python
import soundfile as sf
import voice100_runtime
asr = voice100_runtime.load("asr_en")
waveform, sample_rate = sf.read("output.wav")
text = asr(waveform, sample_rate)
print(text)
```
