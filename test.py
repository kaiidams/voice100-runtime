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
TEST_TEXT_JA_PHONE = (
    "m a t a t o o j i n o y o o n i g o d a i my o: o: t o y o b a r e r u sh u y o: n a my o: o: n o ch u: o: n i"
    "h a i s a r e r u k o t o m o o: i"
)


def abs_length_ratio(x, y):
    lx = len(x)
    ly = len(y)
    return abs(lx - ly) / ly


class Voice100RuntimeTest(unittest.TestCase):
    def test_tts_asr(self):
        tts = voice100_runtime.load("tts_en")
        waveform, sample_rate = tts(TEST_TEXT)
        asr = voice100_runtime.load("asr_en")
        text = asr(waveform, sample_rate)
        self.assertLess(abs_length_ratio(text, TEST_TEXT), 0.05)

    def test_tts(self):
        tts = voice100_runtime.load("tts_en")
        waveform, sample_rate = tts(TEST_TEXT)
        self.assertLess(abs(waveform.shape[0] - 100000) / 100000, 0.05)
        self.assertGreater(waveform.shape[0], 16000)
        self.assertEqual(sample_rate, 16000)
        sf.write("output.wav", waveform, sample_rate, "PCM_16")

    def test_tts_return_align(self):
        tts = voice100_runtime.load("tts_en")
        waveform, sample_rate, aligntext = tts(TEST_TEXT, return_align=True)
        print(aligntext)

    def test_tts_multitask(self):
        tts = voice100_runtime.load("tts_en_mt")
        waveform, sample_rate, phonetext = tts(TEST_TEXT, return_align=True)
        sf.write("output_multitask_en.wav", waveform, sample_rate, "PCM_16")
        print(phonetext)

    def test_asr_ja(self):
        asr = voice100_runtime.load("asr_ja")
        waveform, sample_rate = sf.read("output_ja.wav")
        text = asr(waveform, sample_rate)
        print(text)

    def test_tts_ja(self):
        tts = voice100_runtime.load("tts_ja")
        waveform, sample_rate = tts(TEST_TEXT_JA)
        sf.write("output_ja.wav", waveform, sample_rate, "PCM_16")

    def test_tts_asr_en_v2(self):
        tts = voice100_runtime.load("tts_en_v2")
        waveform, sample_rate = tts(TEST_TEXT)
        asr = voice100_runtime.load("asr_en_v2")
        text = asr(waveform, sample_rate)
        self.assertLess(abs_length_ratio(text, TEST_TEXT), 0.05)

    def test_tts_asr_en_phone_v2(self):
        tts = voice100_runtime.load("tts_en_phone_v2")
        waveform, sample_rate = tts(TEST_TEXT)
        asr = voice100_runtime.load("asr_en_phone_v2")
        _ = asr(waveform, sample_rate)

    def test_asr_ja_v2(self):
        asr = voice100_runtime.load("asr_ja_phone_v2")
        waveform, sample_rate = sf.read("output_ja.wav")
        text = asr(waveform, sample_rate)
        print(text)

    def test_tts_ja_v2(self):
        tts = voice100_runtime.load("tts_ja_phone_v2")
        waveform, sample_rate = tts(TEST_TEXT_JA_PHONE)
        sf.write("output_ja.wav", waveform, sample_rate, "PCM_16")

    def test_list(self):
        models = voice100_runtime.list_models()
        self.assertIn("asr_en", models)
        self.assertIn("tts_en", models)
        self.assertIn("tts_en_phone", models)
        self.assertEqual(len(models), 13)

    def test_load_all(self):
        for model in voice100_runtime.list_models():
            _ = voice100_runtime.load(model)


if __name__ == "__main__":
    unittest.main()
