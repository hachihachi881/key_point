import csv
import glob
import math
import os
from pathlib import Path

import cv2
import numpy as np
import torch
from ultralytics import YOLO


KEYPOINT_NAMES = [
    "nose",
    "left_eye",
    "right_eye",
    "left_ear",
    "right_ear",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
]


MIN_KEYPOINT_CONF = 0.5


def process_video_yolov8(
    video_path,
    output_path=None,
    csv_path=None,
    model_name="yolov8x-pose.pt",
    person_conf=0.8,
    min_person_rows=30,
):
    """
    Run YOLOv8 pose estimation and save an annotated video plus keypoint CSV files.

    CSV layout:
        Each video gets a *_keypoints folder.
        Each person_id is written to a separate CSV named {video_name}{id}.csv in that folder.
        time,nose_x,nose_y,left_eye_x,left_eye_y,...
        0.000000,123.456789,45.678901, ...

    Person CSV files with fewer than min_person_rows data rows are deleted as likely false detections.
    """
    try:
        model = YOLO(model_name)
    except Exception as e:
        print(f"Model load error: {e}")
        print("If YOLOv8 is not installed, run: pip install ultralytics")
        return
    device = 0 if torch.cuda.is_available() else "cpu"

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: could not open video: {video_path}")
        return

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if fps == 0:
        fps = 30

    if output_path is None:
        output_dir = "output_yolov8"
        os.makedirs(output_dir, exist_ok=True)
        video_name = Path(video_path).stem
        output_path = os.path.join(output_dir, f"{video_name}_yolov8.mp4")
    else:
        output_dir = os.path.dirname(output_path) or "."
        os.makedirs(output_dir, exist_ok=True)

    if csv_path is None:
        video_name = Path(video_path).stem
        csv_path = os.path.join(output_dir, f"{video_name}_keypoints")
    else:
        csv_path = str(Path(csv_path).with_suffix(""))

    os.makedirs(csv_path, exist_ok=True)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))
    if not out.isOpened():
        print(f"Error: could not create video: {output_path}")
        cap.release()
        return

    print(f"Processing: {video_path}")
    print(f"Size: {frame_width}x{frame_height}, FPS: {fps}, Frames: {total_frames}")
    print(f"Video output: {output_path}")
    print(f"CSV output folder: {csv_path}")
    print(f"CSV output pattern: {get_person_csv_path(csv_path, '<person_id>')}")
    print(f"Device: {'cuda:0' if device == 0 else 'cpu'}")

    frame_count = 0
    csv_files = {}
    writers = {}
    posture_csv_files = {}
    posture_writers = {}
    person_row_counts = {}

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            results = model(frame, imgsz=480, device=device, conf=person_conf, verbose=False)
            result = results[0]

            write_keypoints_to_csv_files(
                csv_path,
                csv_files,
                writers,
                person_row_counts,
                result,
                frame_count,
                fps,
            )

            posture_results = write_posture_to_csv_files(
                csv_path,
                posture_csv_files,
                posture_writers,
                result,
                frame_count,
                fps,
            )

            annotated_frame = result.plot()
            draw_posture_overlay(annotated_frame, posture_results)
            out.write(annotated_frame)
            frame_count += 1

            if frame_count % 30 == 0:
                progress = (frame_count / total_frames) * 100 if total_frames else 0
                print(f"Progress: {frame_count}/{total_frames} ({progress:.1f}%)")
    finally:
        for csv_file in csv_files.values():
            csv_file.close()
        for csv_file in posture_csv_files.values():
            csv_file.close()

    deleted_person_ids = delete_short_person_csv_files(csv_path, person_row_counts, min_person_rows)
    delete_short_posture_csv_files(csv_path, deleted_person_ids)

    cap.release()
    out.release()

    print(f"Done: {output_path}")
    if csv_files:
        print("CSV saved:")
        for person_id in sorted(csv_files):
            if person_id in deleted_person_ids:
                continue
            print(f"  person_{person_id}: {get_person_csv_path(csv_path, person_id)}")
            posture_csv_path = get_person_posture_csv_path(csv_path, person_id)
            if Path(posture_csv_path).exists():
                print(f"    posture: {posture_csv_path}")
        if deleted_person_ids:
            print(f"Deleted short CSV files: {len(deleted_person_ids)}")
            for person_id in deleted_person_ids:
                row_count = person_row_counts.get(person_id, 0)
                print(f"  person_{person_id}: {row_count} rows (< {min_person_rows})")
    else:
        print("CSV saved: no persons detected, so no person CSV files were created")


def build_csv_header():
    header = ["time"]

    for keypoint_name in KEYPOINT_NAMES:
        header.extend([f"{keypoint_name}_x", f"{keypoint_name}_y", f"{keypoint_name}_conf"])

    return header


def build_posture_csv_header():
    return [
        "time",
        "side",
        "head_forward_ratio",
        "trunk_angle",
        "trunk_angle_left",
        "trunk_angle_right",
        "trunk_angle_opposite",
        "trunk_angle_mid",
        "head_forward_score",
        "trunk_forward_score",
        "bad_reasons",
    ]


def get_person_csv_path(csv_path, person_id):
    path = Path(csv_path)
    video_name = path.name
    if video_name.endswith("_keypoints"):
        video_name = video_name[: -len("_keypoints")]

    return str(path / f"{video_name}{person_id}.csv")


def get_person_posture_csv_path(csv_path, person_id):
    path = Path(csv_path)
    video_name = path.name
    if video_name.endswith("_keypoints"):
        video_name = video_name[: -len("_keypoints")]

    return str(path / f"{video_name}{person_id}_posture.csv")


def get_person_csv_writer(csv_path, csv_files, writers, person_id):
    if person_id not in writers:
        person_csv_path = get_person_csv_path(csv_path, person_id)
        csv_file = open(person_csv_path, "w", newline="", encoding="utf-8-sig")
        writer = csv.writer(csv_file)
        writer.writerow(build_csv_header())
        csv_files[person_id] = csv_file
        writers[person_id] = writer

    return writers[person_id]


def get_person_posture_csv_writer(csv_path, csv_files, writers, person_id):
    if person_id not in writers:
        person_csv_path = get_person_posture_csv_path(csv_path, person_id)
        csv_file = open(person_csv_path, "w", newline="", encoding="utf-8-sig")
        writer = csv.writer(csv_file)
        writer.writerow(build_posture_csv_header())
        csv_files[person_id] = csv_file
        writers[person_id] = writer

    return writers[person_id]


def write_keypoints_to_csv_files(csv_path, csv_files, writers, person_row_counts, result, frame_count, fps):
    if result.keypoints is None or result.keypoints.xy is None:
        return

    keypoints_xy = result.keypoints.xy.cpu().numpy()
    keypoints_conf = get_keypoints_conf(result, len(keypoints_xy))

    time_sec = frame_count / fps
    for person_id, person_keypoints in enumerate(keypoints_xy):
        writer = get_person_csv_writer(csv_path, csv_files, writers, person_id)
        row = [round(time_sec, 6)]

        for keypoint_id in range(len(KEYPOINT_NAMES)):
            if keypoint_id < len(person_keypoints):
                x, y = person_keypoints[keypoint_id]
                conf = keypoints_conf[person_id][keypoint_id] if keypoints_conf is not None else np.nan
                row.extend([round(float(x), 6), round(float(y), 6), round_or_blank(conf)])
            else:
                row.extend(["", "", ""])

        writer.writerow(row)
        person_row_counts[person_id] = person_row_counts.get(person_id, 0) + 1


def write_posture_to_csv_files(csv_path, csv_files, writers, result, frame_count, fps):
    posture_results = []
    if result.keypoints is None or result.keypoints.xy is None:
        return posture_results

    keypoints_xy = result.keypoints.xy.cpu().numpy()
    keypoints_conf = get_keypoints_conf(result, len(keypoints_xy))

    time_sec = frame_count / fps
    for person_id, person_keypoints in enumerate(keypoints_xy):
        person_conf = keypoints_conf[person_id] if keypoints_conf is not None else None
        assessment = assess_side_view_posture(person_keypoints, person_conf)
        posture_results.append((person_id, assessment))

        writer = get_person_posture_csv_writer(csv_path, csv_files, writers, person_id)
        writer.writerow(
            [
                round(time_sec, 6),
                assessment["side"],
                round_or_blank(assessment["head_forward_ratio"]),
                round_or_blank(assessment["trunk_angle"]),
                round_or_blank(assessment["trunk_angle_left"]),
                round_or_blank(assessment["trunk_angle_right"]),
                round_or_blank(assessment["trunk_angle_opposite"]),
                round_or_blank(assessment["trunk_angle_mid"]),
                round_or_blank(assessment["head_forward_score"], digits=1),
                round_or_blank(assessment["trunk_forward_score"], digits=1),
                ";".join(assessment["bad_reasons"]),
            ]
        )

    return posture_results


def get_keypoints_conf(result, person_count):
    if result.keypoints.conf is None:
        return None

    keypoints_conf = result.keypoints.conf.cpu().numpy()
    if len(keypoints_conf) != person_count:
        return None

    return keypoints_conf


def assess_side_view_posture(person_keypoints, person_conf=None, min_conf=MIN_KEYPOINT_CONF):
    front_side = choose_front_side(person_keypoints, person_conf)
    point = make_point_reader(person_keypoints, person_conf, min_conf)

    ear = point(f"{front_side}_ear")
    shoulder = point(f"{front_side}_shoulder")
    hip = point(f"{front_side}_hip")

    head_forward_ratio = head_forward_ratio_from_torso_line(ear, shoulder, hip)

    trunk_angles = compare_trunk_angles(point, front_side)
    trunk_angle = trunk_angles["front"]

    head_score, head_reason = score_head_forward(head_forward_ratio)
    trunk_score, trunk_reason = score_trunk_angle(trunk_angle)

    bad_reasons = [
        reason
        for reason in [head_reason, trunk_reason]
        if reason
    ]

    return {
        "side": front_side,
        "head_forward_ratio": head_forward_ratio,
        "trunk_angle": trunk_angle,
        "trunk_angle_left": trunk_angles["left"],
        "trunk_angle_right": trunk_angles["right"],
        "trunk_angle_opposite": trunk_angles["back"],
        "trunk_angle_mid": trunk_angles["mid"],
        "head_forward_score": head_score,
        "trunk_forward_score": trunk_score,
        "bad_reasons": bad_reasons,
        "anchor": first_available_point(shoulder, hip, ear),
    }


def compare_trunk_angles(point, front_side):
    back_side = "right" if front_side == "left" else "left"

    left_shoulder = point("left_shoulder")
    left_hip = point("left_hip")
    right_shoulder = point("right_shoulder")
    right_hip = point("right_hip")

    left_angle = angle_from_vertical(left_shoulder, left_hip)
    right_angle = angle_from_vertical(right_shoulder, right_hip)
    front_angle = left_angle if front_side == "left" else right_angle
    back_angle = right_angle if back_side == "right" else left_angle

    shoulder_mid = midpoint(left_shoulder, right_shoulder)
    hip_mid = midpoint(left_hip, right_hip)
    mid_angle = angle_from_vertical(shoulder_mid, hip_mid)

    return {
        "front": front_angle,
        "left": left_angle,
        "right": right_angle,
        "back": back_angle,
        "mid": mid_angle,
    }


def choose_front_side(person_keypoints, person_conf):
    """Choose the camera-side body side from visible ear, shoulder, and hip confidence."""
    side_scores = {}
    for side in ["left", "right"]:
        ids = [
            KEYPOINT_NAMES.index(f"{side}_ear"),
            KEYPOINT_NAMES.index(f"{side}_shoulder"),
            KEYPOINT_NAMES.index(f"{side}_hip"),
        ]
        score = 0.0
        for keypoint_id in ids:
            if keypoint_id >= len(person_keypoints):
                continue

            x, y = person_keypoints[keypoint_id]
            if float(x) == 0.0 and float(y) == 0.0:
                continue

            if person_conf is None:
                score += 1.0
            else:
                score += float(person_conf[keypoint_id])

        side_scores[side] = score

    return max(side_scores, key=side_scores.get)


def first_available_point(*points):
    for point in points:
        if point is not None:
            return point
    return None


def midpoint(point_a, point_b):
    if point_a is None or point_b is None:
        return None
    return (point_a + point_b) / 2.0


def make_point_reader(person_keypoints, person_conf, min_conf):
    def read_point(name):
        keypoint_id = KEYPOINT_NAMES.index(name)
        if keypoint_id >= len(person_keypoints):
            return None
        if person_conf is not None and float(person_conf[keypoint_id]) < min_conf:
            return None

        x, y = person_keypoints[keypoint_id]
        if float(x) == 0.0 and float(y) == 0.0:
            return None
        return np.array([float(x), float(y)], dtype=float)

    return read_point


def distance(point_a, point_b):
    if point_a is None or point_b is None:
        return None
    return float(np.linalg.norm(point_a - point_b))


def head_forward_ratio_from_torso_line(ear, shoulder, hip):
    if ear is None or shoulder is None or hip is None:
        return None

    torso = shoulder - hip
    torso_length = np.linalg.norm(torso)
    if torso_length == 0:
        return None

    ear_offset = ear - shoulder
    cross = torso[0] * ear_offset[1] - torso[1] * ear_offset[0]
    perpendicular_distance = abs(float(cross)) / torso_length
    return perpendicular_distance / torso_length


def angle_from_vertical(top_point, bottom_point):
    if top_point is None or bottom_point is None:
        return None

    vector = top_point - bottom_point
    norm = np.linalg.norm(vector)
    if norm == 0:
        return None

    vertical = np.array([0.0, -1.0])
    cos_angle = float(np.dot(vector, vertical) / norm)
    cos_angle = abs(max(-1.0, min(1.0, cos_angle)))
    return math.degrees(math.acos(cos_angle))


def score_head_forward(value):
    if value is None:
        return None, "head_unavailable"

    score = piecewise_score(
        value,
        [
            (0.00, 100.0),
            (0.04, 85.0),
            (0.10, 50.0),
            (0.18, 0.0),
        ],
    )
    reason = "head_forward" if score < 85.0 else ""
    return score, reason


def score_trunk_angle(value):
    if value is None:
        return None, "trunk_unavailable"

    score = piecewise_score(
        value,
        [
            (0.0, 100.0),
            (5.0, 85.0),
            (15.0, 50.0),
            (30.0, 0.0),
        ],
    )
    reason = "trunk_forward" if score < 85.0 else ""
    return score, reason


def piecewise_score(value, points):
    if value <= points[0][0]:
        return points[0][1]

    for (x_min, y_min), (x_max, y_max) in zip(points, points[1:]):
        if value <= x_max:
            return interpolate_score(value, x_min, x_max, y_min, y_max)

    return points[-1][1]


def interpolate_score(value, x_min, x_max, y_min, y_max):
    if x_max == x_min:
        return y_max
    ratio = (value - x_min) / (x_max - x_min)
    ratio = max(0.0, min(1.0, ratio))
    return y_min + ratio * (y_max - y_min)


def draw_posture_overlay(frame, posture_results):
    for person_id, assessment in posture_results:
        head_score = assessment["head_forward_score"]
        trunk_score = assessment["trunk_forward_score"]
        if head_score is None and trunk_score is None:
            text = f"id {person_id}: posture unavailable"
            color = (180, 180, 180)
        else:
            head_text = format_score_for_overlay(head_score)
            trunk_text = format_score_for_overlay(trunk_score)
            text = f"id {person_id}: head {head_text} / trunk {trunk_text}"
            color = posture_color(min_available_score(head_score, trunk_score))

        anchor = assessment.get("anchor")
        if anchor is None:
            x, y = 20, 30 + person_id * 28
        else:
            x = int(anchor[0] + 10)
            y = max(25, int(anchor[1] - 10))

        cv2.putText(frame, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(frame, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA)


def format_score_for_overlay(score):
    if score is None:
        return "NA"
    return f"{score:.0f}"


def min_available_score(*scores):
    available_scores = [score for score in scores if score is not None]
    if not available_scores:
        return None
    return min(available_scores)


def posture_color(score):
    if score is None:
        return (180, 180, 180)
    if score >= 85:
        return (0, 180, 0)
    if score >= 70:
        return (0, 200, 200)
    if score >= 50:
        return (0, 165, 255)
    return (0, 0, 255)


def round_or_blank(value, digits=6):
    if value is None:
        return ""
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return ""
    return round(float(value), digits)


def delete_short_person_csv_files(csv_path, person_row_counts, min_person_rows):
    deleted_person_ids = []

    if min_person_rows <= 0:
        return deleted_person_ids

    for person_id, row_count in sorted(person_row_counts.items()):
        if row_count >= min_person_rows:
            continue

        person_csv_path = Path(get_person_csv_path(csv_path, person_id))
        if person_csv_path.exists():
            person_csv_path.unlink()
            deleted_person_ids.append(person_id)

    return deleted_person_ids


def delete_short_posture_csv_files(csv_path, person_ids):
    for person_id in person_ids:
        person_csv_path = Path(get_person_posture_csv_path(csv_path, person_id))
        if person_csv_path.exists():
            person_csv_path.unlink()


if __name__ == "__main__":
    video_dir = "input_movie"

    if not os.path.exists(video_dir):
        print(f"Error: folder not found: {video_dir}")
        exit(1)

    video_extensions = ["*.mp4", "*.avi", "*.mov", "*.mkv"]

    video_files = []
    for ext in video_extensions:
        video_files.extend(glob.glob(os.path.join(video_dir, ext)))

    if not video_files:
        print(f"Error: no video files found in: {video_dir}")
        exit(1)

    print(f"Found video files: {len(video_files)}")
    print("Model: yolov8x-pose.pt")

    for video_path in video_files:
        print("\n" + "=" * 60)
        process_video_yolov8(video_path)

    print("\n" + "=" * 60)
    print("All processing completed")
