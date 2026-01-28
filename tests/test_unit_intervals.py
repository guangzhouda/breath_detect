import unittest

from breath_detect_rule import invert_intervals


class TestInvertIntervals(unittest.TestCase):
    def test_invert_basic(self):
        segments = [(1.0, 2.0), (3.0, 4.0)]
        non_speech = invert_intervals(segments, total_duration_s=5.0)
        self.assertEqual(non_speech, [(0.0, 1.0), (2.0, 3.0), (4.0, 5.0)])


if __name__ == "__main__":
    unittest.main()
