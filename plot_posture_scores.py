"""Plot head and trunk posture scores from a posture CSV file."""

import argparse
import csv
import math
from pathlib import Path


REQUIRED_COLUMNS = {"time", "head_forward_score", "trunk_forward_score"}


def read_posture_scores(csv_path):
    """Read time, head score, and trunk score while preserving missing-value gaps."""
    csv_path = Path(csv_path)
    times = []
    head_scores = []
    trunk_scores = []

    with csv_path.open("r", encoding="utf-8-sig", newline="") as csv_file:
        reader = csv.DictReader(csv_file)
        columns = set(reader.fieldnames or [])
        missing_columns = sorted(REQUIRED_COLUMNS - columns)
        if missing_columns:
            raise ValueError(f"Missing required columns: {', '.join(missing_columns)}")

        for row in reader:
            time_value = parse_number(row.get("time"))
            if math.isnan(time_value):
                continue
            times.append(time_value)
            head_scores.append(parse_number(row.get("head_forward_score")))
            trunk_scores.append(parse_number(row.get("trunk_forward_score")))

    if not times:
        raise ValueError("The CSV does not contain any valid time rows.")
    if all(math.isnan(value) for value in head_scores + trunk_scores):
        raise ValueError("The CSV does not contain any valid posture scores.")

    return times, head_scores, trunk_scores


def parse_number(value):
    if value is None or not str(value).strip():
        return math.nan
    try:
        number = float(value)
    except (TypeError, ValueError):
        return math.nan
    return number if math.isfinite(number) else math.nan


def default_output_path(csv_path):
    csv_path = Path(csv_path)
    return csv_path.with_name(f"{csv_path.stem}_scores.png")


def plot_posture_scores(csv_path, output_path=None, show=False):
    """Create a two-line score graph and return the saved PNG path."""
    import matplotlib.pyplot as plt

    csv_path = Path(csv_path)
    output_path = Path(output_path) if output_path else default_output_path(csv_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    times, head_scores, trunk_scores = read_posture_scores(csv_path)

    figure, axis = plt.subplots(figsize=(12, 6))
    axis.plot(times, head_scores, color="#1976d2", linewidth=1.6, label="Head forward score")
    axis.plot(times, trunk_scores, color="#ef6c00", linewidth=1.6, label="Trunk score")
    axis.axhline(85, color="#2e7d32", linewidth=1.0, linestyle="--", alpha=0.7, label="85-point threshold")
    axis.set_title(f"Posture score comparison - {csv_path.stem}")
    axis.set_xlabel("Time (seconds)")
    axis.set_ylabel("Score")
    axis.set_ylim(0, 105)
    axis.set_xlim(left=0)
    axis.grid(True, alpha=0.25)
    axis.legend(loc="lower left")
    figure.tight_layout()
    figure.savefig(output_path, dpi=160)

    if show:
        plt.show()
    plt.close(figure)
    return output_path


def choose_csv_file():
    import tkinter as tk
    from tkinter import filedialog

    root = tk.Tk()
    root.withdraw()
    try:
        return filedialog.askopenfilename(
            title="Select a posture CSV",
            filetypes=[("Posture CSV", "*_posture.csv"), ("CSV files", "*.csv"), ("All files", "*.*")],
        )
    finally:
        root.destroy()


def build_argument_parser():
    parser = argparse.ArgumentParser(
        description="Plot head_forward_score and trunk_forward_score from a posture CSV."
    )
    parser.add_argument("csv_path", nargs="?", help="Path to a *_posture.csv file")
    parser.add_argument("-o", "--output", help="Output PNG path")
    parser.add_argument("--show", action="store_true", help="Display the graph after saving it")
    return parser


def main():
    parser = build_argument_parser()
    args = parser.parse_args()
    csv_path = args.csv_path or choose_csv_file()
    if not csv_path:
        print("No CSV file was selected.")
        return 1

    try:
        output_path = plot_posture_scores(csv_path, args.output, args.show)
    except (OSError, ValueError) as exc:
        parser.error(str(exc))

    print(f"Graph saved: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
