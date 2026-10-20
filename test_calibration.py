import unittest
from datetime import datetime
from pathlib import Path
import tempfile

from calibration import CalibrationCollector, CalibrationError, calibrated_value
from output_paths import create_dated_output_paths


class CalibrationCollectorTest(unittest.TestCase):
    def test_median_is_robust_to_outlier(self):
        collector = CalibrationCollector()
        for head, trunk in [(0.08, 3.0), (0.09, 3.2), (0.085, 3.1), (1.0, 45.0)]:
            collector.add({"head_forward_ratio": head, "trunk_angle": trunk})

        baseline = collector.finish(min_valid_frames=3, min_valid_ratio=0.5)

        self.assertAlmostEqual(baseline.head_forward_ratio, 0.0875)
        self.assertAlmostEqual(baseline.trunk_angle, 3.15)
        self.assertEqual(baseline.valid_frames, 4)

    def test_invalid_frames_are_counted_but_not_used(self):
        collector = CalibrationCollector()
        collector.add({"head_forward_ratio": 0.08, "trunk_angle": 3.0})
        collector.add({"head_forward_ratio": None, "trunk_angle": 3.0})
        collector.add({"head_forward_ratio": 0.09, "trunk_angle": None})

        with self.assertRaises(CalibrationError):
            collector.finish(min_valid_frames=1, min_valid_ratio=0.5)

    def test_calibrated_value_only_measures_deterioration(self):
        self.assertAlmostEqual(calibrated_value(0.12, 0.08), 0.04)
        self.assertEqual(calibrated_value(0.06, 0.08), 0.0)
        self.assertIsNone(calibrated_value(None, 0.08))


class OutputPathsTest(unittest.TestCase):
    def test_outputs_are_grouped_by_date_and_do_not_overwrite(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            executed_at = datetime(2026, 10, 20, 15, 30, 45)
            video_path, csv_path = create_dated_output_paths(
                "sample.mp4", temp_dir, now=executed_at
            )

            self.assertEqual(video_path.parent.name, "2026-10-20")
            self.assertEqual(video_path.name, "sample_153045_yolov8.mp4")
            self.assertEqual(csv_path.name, "sample_153045_keypoints")

            Path(video_path).touch()
            second_video_path, second_csv_path = create_dated_output_paths(
                "sample.mp4", temp_dir, now=executed_at
            )
            self.assertEqual(second_video_path.name, "sample_153045_2_yolov8.mp4")
            self.assertEqual(second_csv_path.name, "sample_153045_2_keypoints")

    def test_mode_is_included_for_comparison_outputs(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            executed_at = datetime(2026, 10, 20, 15, 30, 45)
            video_path, csv_path = create_dated_output_paths(
                "sample.mp4",
                temp_dir,
                now=executed_at,
                mode="uncalibrated",
            )

            self.assertEqual(video_path.name, "sample_153045_uncalibrated_yolov8.mp4")
            self.assertEqual(csv_path.name, "sample_153045_uncalibrated_keypoints")


if __name__ == "__main__":
    unittest.main()
