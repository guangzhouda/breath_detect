import unittest
import tempfile

import numpy as np
import soundfile as sf

from breath_detect_rule import BreathRuleConfig, VadConfig, detect_breath_with_vad, _load_silero_vad


class TestFunctionalVad(unittest.TestCase):
    def test_vad_pipeline(self):
        # 功能测试依赖 silero-vad + torch，且需要文件路径输入（silero_vad.read_audio）
        vad_repo = "E:\\Projects\\silero-vad"

        try:
            _load_silero_vad(str(vad_repo))
        except Exception as exc:
            self.skipTest(f"silero-vad 不可用或依赖缺失: {exc}")

        sr = 16000
        rng = np.random.default_rng(0)
        x = (0.001 * rng.standard_normal(sr)).astype(np.float32)

        with tempfile.TemporaryDirectory() as td:
            wav_path = td + "\\tmp.wav"
            sf.write(wav_path, x, sr)
            cfg = BreathRuleConfig(debug=False)
            vad_cfg = VadConfig(vad_repo=str(vad_repo), return_seconds=True)
            out = detect_breath_with_vad(str(wav_path), cfg, vad_cfg)
            self.assertIn("non_speech_segments", out)
            self.assertIn("breath_segments", out)


if __name__ == "__main__":
    unittest.main()
