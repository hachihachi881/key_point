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


def default_comparison_output_path(first_csv_path, second_csv_path):
    first_csv_path = Path(first_csv_path)
    return first_csv_path.with_name(f"{first_csv_path.stem}_calibration_comparison.png")


def infer_condition_label(csv_path):
    """Infer Calibrated/Uncalibrated from calibration_state or the file name."""
    csv_path = Path(csv_path)
    with csv_path.open("r", encoding="utf-8-sig", newline="") as csv_file:
        reader = csv.DictReader(csv_file)
        if "calibration_state" in (reader.fieldnames or []):
            for row in reader:
                state = (row.get("calibration_state") or "").strip().lower()
                if state == "calibrated":
                    return "Calibrated"
                if state in {"disabled", "uncalibrated"}:
                    return "Uncalibrated"

    lower_name = csv_path.name.lower()
    if "uncalibrated" in lower_name:
        return "Uncalibrated"
    if "calibrated" in lower_name:
        return "Calibrated"
    return csv_path.stem


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


def plot_calibration_comparison(
    first_csv_path,
    second_csv_path,
    output_path=None,
    show=False,
    first_label=None,
    second_label=None,
):
    """Compare calibrated and uncalibrated score CSV files in two aligned panels."""
    import matplotlib.pyplot as plt

    first_csv_path = Path(first_csv_path)
    second_csv_path = Path(second_csv_path)
    output_path = (
        Path(output_path)
        if output_path
        else default_comparison_output_path(first_csv_path, second_csv_path)
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)

    first_time, first_head, first_trunk = read_posture_scores(first_csv_path)
    second_time, second_head, second_trunk = read_posture_scores(second_csv_path)
    first_label = first_label or infer_condition_label(first_csv_path)
    second_label = second_label or infer_condition_label(second_csv_path)

    figure, axes = plt.subplots(2, 1, figsize=(12, 9), sharex=True, sharey=True)
    panels = [
        (axes[0], "Head forward score", first_head, second_head),
        (axes[1], "Trunk score", first_trunk, second_trunk),
    ]

    for axis, title, first_scores, second_scores in panels:
        axis.plot(first_time, first_scores, color="#1976d2", linewidth=1.5, label=first_label)
        axis.plot(second_time, second_scores, color="#ef6c00", linewidth=1.5, label=second_label)
        axis.axhline(85, color="#2e7d32", linewidth=1.0, linestyle="--", alpha=0.7, label="85-point threshold")
        axis.set_title(title)
        axis.set_ylabel("Score")
        axis.set_ylim(0, 105)
        axis.set_xlim(left=0)
        axis.grid(True, alpha=0.25)
        axis.legend(loc="lower left")

    axes[1].set_xlabel("Time (seconds)")
    figure.suptitle("Calibrated vs. uncalibrated posture scores")
    figure.tight_layout()
    figure.savefig(output_path, dpi=160)

    if show:
        plt.show()
    plt.close(figure)
    return output_path


def choose_csv_files():
    import tkinter as tk
    from tkinter import filedialog

    root = tk.Tk()
    root.withdraw()
    try:
        return filedialog.askopenfilenames(
            title="Select calibrated and uncalibrated posture CSV files",
            filetypes=[("Posture CSV", "*_posture.csv"), ("CSV files", "*.csv"), ("All files", "*.*")],
        )
    finally:
        root.destroy()


def build_argument_parser():
    parser = argparse.ArgumentParser(
        description="Plot head_forward_score and trunk_forward_score from a posture CSV."
    )
    parser.add_argument(
        "csv_paths",
        nargs="*",
        help="One posture CSV, or calibrated and uncalibrated posture CSV files",
    )
    parser.add_argument("-o", "--output", help="Output PNG path")
    parser.add_argument("--show", action="store_true", help="Display the graph after saving it")
    parser.add_argument("--label-a", help="Legend label for the first CSV")
    parser.add_argument("--label-b", help="Legend label for the second CSV")
    return parser


def main():
    parser = build_argument_parser()
    args = parser.parse_args()
    csv_paths = args.csv_paths or list(choose_csv_files())
    if not csv_paths:
        print("No CSV files were selected.")
        return 1

    try:
        if len(csv_paths) == 1:
            output_path = plot_posture_scores(csv_paths[0], args.output, args.show)
        elif len(csv_paths) == 2:
            output_path = plot_calibration_comparison(
                csv_paths[0],
                csv_paths[1],
                args.output,
                args.show,
                args.label_a,
                args.label_b,
            )
        else:
            parser.error("Select exactly one CSV for a single graph or two CSV files for comparison.")
    except (OSError, ValueError) as exc:
        parser.error(str(exc))

    print(f"Graph saved: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
