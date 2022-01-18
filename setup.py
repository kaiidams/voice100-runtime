from setuptools import setup

setup(
    name="voice100-runtime",
    version="1.1.0",
    author="Katsuya Iida",
    author_email="katsuya.iida@gmail.com",
    description="Voice100 Runtime",
    license="MIT",
    url="https://github.com/kaiidams/voice100-runtime",
    packages=["voice100_runtime"],
    long_description="""Voice100 Runtime is a TTS/ASR sample app that uses
ONNX Runtime, WORLD and Voice100 neural TTS/ASR models on Python.
Inference of Voice100 is low cost as its models are tiny and only depend
on CNN without recursion.""",
    entry_points={},
    install_requires=["onnxruntime", "librosa", "numpy", "pyworld"],
    extras_require={
        "lang-en_phone": [
            "g2p-en"
        ]
    })
