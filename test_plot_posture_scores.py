import csv
from pathlib import Path
import tempfile
import unittest

from plot_posture_scores import default_output_path, plot_posture_scores, read_posture_scores


class PlotPostureScoresTest(unittest.TestCase):
    def write_csv(self, directory, fieldnames, rows):
        path = Path(directory) / "sample_posture.csv"
        with path.open("w", encoding="utf-8-sig", newline="") as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        return path

    def test_reads_two_scores_and_preserves_missing_values(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            csv_path = self.write_csv(
                temp_dir,
                ["time", "head_forward_score", "trunk_forward_score"],
                [
                    {"time": "0.0", "head_forward_score": "100", "trunk_forward_score": "90"},
                    {"time": "0.1", "head_forward_score": "", "trunk_forward_score": "80"},
                ],
            )

            times, head_scores, trunk_scores = read_posture_scores(csv_path)

            self.assertEqual(times, [0.0, 0.1])
            self.assertEqual(head_scores[0], 100.0)
            self.assertTrue(head_scores[1] != head_scores[1])
            self.assertEqual(trunk_scores, [90.0, 80.0])

    def test_rejects_csv_without_required_columns(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            csv_path = self.write_csv(temp_dir, ["time", "head_forward_score"], [])
            with self.assertRaisesRegex(ValueError, "trunk_forward_score"):
                read_posture_scores(csv_path)

    def test_creates_png_next_to_csv(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            csv_path = self.write_csv(
                temp_dir,
                ["time", "head_forward_score", "trunk_forward_score"],
                [
                    {"time": "0.0", "head_forward_score": "100", "trunk_forward_score": "100"},
                    {"time": "1.0", "head_forward_score": "75", "trunk_forward_score": "60"},
                ],
            )

            output_path = plot_posture_scores(csv_path)

            self.assertEqual(output_path, default_output_path(csv_path))
            self.assertTrue(output_path.is_file())
            self.assertGreater(output_path.stat().st_size, 0)


if __name__ == "__main__":
    unittest.main()
