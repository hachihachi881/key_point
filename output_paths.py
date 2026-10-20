"""Output path helpers shared by offline and real-time processing."""

from datetime import datetime
from pathlib import Path


def create_dated_output_paths(video_path, output_root="output_yolov8", now=None):
    """Create unique video/CSV paths under an execution-date directory."""
    executed_at = now or datetime.now().astimezone()
    date_dir = Path(output_root) / executed_at.strftime("%Y-%m-%d")
    date_dir.mkdir(parents=True, exist_ok=True)

    source_name = Path(video_path).stem
    base_name = f"{source_name}_{executed_at.strftime('%H%M%S')}"
    candidate = base_name
    suffix = 2

    while (date_dir / f"{candidate}_yolov8.mp4").exists() or (date_dir / f"{candidate}_keypoints").exists():
        candidate = f"{base_name}_{suffix}"
        suffix += 1

    return (
        date_dir / f"{candidate}_yolov8.mp4",
        date_dir / f"{candidate}_keypoints",
    )


def csv_path_for_output_video(output_path):
    """Derive a non-conflicting CSV directory from an explicitly named video."""
    output_path = Path(output_path)
    run_name = output_path.stem
    if run_name.endswith("_yolov8"):
        run_name = run_name[: -len("_yolov8")]
    return output_path.parent / f"{run_name}_keypoints"
