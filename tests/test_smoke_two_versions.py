import unittest
import numpy as np


class TestSmokeTwoVersions(unittest.TestCase):
    def test_zff_no_vad_runs(self):
        from breath_detect_two_versions import read_audio_mono, BreathFeatCfg, ZFFCfg, breath_detect_no_vad_with_zff

        rng = np.random.default_rng(0)
        sr = 16000
        x = (0.001 * rng.standard_normal(sr)).astype(np.float32)

        feat_cfg = BreathFeatCfg(sr=sr)
        zff_cfg = ZFFCfg(sr=sr)
        segs = breath_detect_no_vad_with_zff(x, zff_cfg, feat_cfg)
        self.assertIsInstance(segs, list)

    def test_vad_no_zff_window_logic_runs(self):
        from breath_detect_two_versions import BreathFeatCfg, VADCfg, breath_detect_vad_no_zff

        rng = np.random.default_rng(0)
        sr = 16000
        x = (0.001 * rng.standard_normal(sr)).astype(np.float32)

        # 用伪造语音区间测试窗口逻辑（避免依赖 webrtcvad / silero-vad）
        speech_intervals = [(int(0.2 * sr), int(0.5 * sr))]
        feat_cfg = BreathFeatCfg(sr=sr)
        vad_cfg = VADCfg(sr=sr, window_mode="non_speech")
        segs = breath_detect_vad_no_zff(x, vad_cfg, feat_cfg, speech_intervals=speech_intervals)
        self.assertIsInstance(segs, list)


if __name__ == "__main__":
    unittest.main()
