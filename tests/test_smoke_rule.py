import unittest

import numpy as np

from breath_detect_rule import BreathRuleConfig, detect_breath_rule_from_array


class TestSmokeRule(unittest.TestCase):
    def test_rule_pipeline_runs(self):
        rng = np.random.default_rng(0)
        sr = 16000
        # 生成 1 秒的可运行样本（无需依赖仓库内 wav 文件）
        x = (0.001 * rng.standard_normal(sr)).astype(np.float32)
        cfg = BreathRuleConfig(debug=False)
        out = detect_breath_rule_from_array(x, sr, cfg)
        self.assertIn("segments", out)
        self.assertIn("coverage", out)
        self.assertIn("duration_s", out)


if __name__ == "__main__":
    unittest.main()
