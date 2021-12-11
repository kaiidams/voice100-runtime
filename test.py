import unittest
import soundfile as sf
import voice100_runtime

TEST_TEXT = (
    "Beginnings are apt to be determinative and when reinforced"
    " by continuous applications of similar influence."
)
TEST_TEXT_JA = (
    "mata toojinoyoonigodaimyooootoyobarerushuyoonamyoooonochuuooni"
    "haisarerukotomoooi"
)


class Voice100RuntimeTest(unittest.TestCase):
    def test_stt(self):
        stt = voice100_runtime.load("stt_en")
        waveform, sample_rate = sf.read("output.wav")
        text = stt(waveform, sample_rate)
        print(text)

    def test_tts(self):
        tts = voice100_runtime.load("tts_en")
        waveform, sample_rate = tts(TEST_TEXT)
        sf.write("output.wav", waveform, sample_rate, "PCM_16")

    def test_stt_ja(self):
        stt = voice100_runtime.load("stt_ja")
        waveform, sample_rate = sf.read("output_ja.wav")
        text = stt(waveform, sample_rate)
        print(text)

    def test_tts_ja(self):
        tts = voice100_runtime.load("tts_ja")
        waveform, sample_rate = tts(TEST_TEXT_JA)
        sf.write("output_ja.wav", waveform, sample_rate, "PCM_16")


if __name__ == "__main__":
    unittest.main()
