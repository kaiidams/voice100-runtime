from setuptools import setup

setup(
    name="voice100-runtime",
    version="0.1",
    author="Katsuya Iida",
    author_email="katsuya.iida@gmail.com",
    description="Voice100 runtime",
    license="MIT",
    url="https://github.com/kaiidams/voice100-runtime",
    packages=['voice100_runtime'],
    long_description="Voice100 is a small TTS.",
    entry_points={
    },
    install_requires=[
        'onnxruntime',
        'librosa',
        'numpy',
        'pyworld'
    ])
