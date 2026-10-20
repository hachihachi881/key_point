"""Desktop launcher and interactive calibration UI for posture assessment."""

import os
from pathlib import Path
import time
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import cv2
import torch
from ultralytics import YOLO

from calibration import CalibrationCollector, CalibrationError
from output_paths import create_dated_output_paths
from process_video_yolov8 import assess_side_view_posture, get_keypoints_conf, process_video_yolov8


WINDOW_NAME = "Posture calibration"
VIDEO_TYPES = [("Video files", "*.mp4 *.avi *.mov *.mkv"), ("All files", "*.*")]


def first_person_assessment(result):
    if result.keypoints is None or result.keypoints.xy is None:
        return None

    keypoints_xy = result.keypoints.xy.cpu().numpy()
    if len(keypoints_xy) == 0:
        return None

    keypoints_conf = get_keypoints_conf(result, len(keypoints_xy))
    person_conf = keypoints_conf[0] if keypoints_conf is not None else None
    return assess_side_view_posture(keypoints_xy[0], person_conf)


def draw_panel(frame, lines, color=(255, 255, 255)):
    overlay = frame.copy()
    panel_height = 54 + 32 * len(lines)
    cv2.rectangle(overlay, (0, 0), (frame.shape[1], panel_height), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.72, frame, 0.28, 0, frame)
    for index, line in enumerate(lines):
        cv2.putText(
            frame,
            line,
            (24, 42 + index * 32),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.72,
            color,
            2,
            cv2.LINE_AA,
        )


def draw_button(frame, label, rect, enabled=True):
    x1, y1, x2, y2 = rect
    fill = (50, 135, 60) if enabled else (90, 90, 90)
    cv2.rectangle(frame, (x1, y1), (x2, y2), fill, -1)
    cv2.rectangle(frame, (x1, y1), (x2, y2), (240, 240, 240), 2)
    text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 2)[0]
    text_x = x1 + max(8, (x2 - x1 - text_size[0]) // 2)
    text_y = y1 + (y2 - y1 + text_size[1]) // 2
    cv2.putText(frame, label, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)


def point_in_rect(x, y, rect):
    x1, y1, x2, y2 = rect
    return x1 <= x <= x2 and y1 <= y <= y2


def fit_for_display(frame, max_width=1280, max_height=800):
    height, width = frame.shape[:2]
    scale = min(max_width / width, max_height / height, 1.0)
    if scale == 1.0:
        return frame
    return cv2.resize(frame, (int(width * scale), int(height * scale)))


def run_calibration_session(
    video_path,
    model_name,
    duration_seconds=5.0,
    countdown_seconds=3.0,
    person_conf=0.8,
):
    """Interactively select and measure a good-posture interval."""
    model = YOLO(model_name)
    device = 0 if torch.cuda.is_available() else "cpu"
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")

    fps = float(cap.get(cv2.CAP_PROP_FPS)) or 30.0
    calibration_frames = max(1, int(round(duration_seconds * fps)))
    state = {"name": "preview", "action": None, "buttons": {}}
    collector = CalibrationCollector()
    baseline = None
    error_message = ""
    frames_left = calibration_frames
    countdown_deadline = None
    last_annotated = None
    last_detection_ok = False

    def on_mouse(event, x, y, flags, param):
        if event != cv2.EVENT_LBUTTONDOWN:
            return
        for action, rect in state["buttons"].items():
            if point_in_rect(x, y, rect):
                state["action"] = action
                return

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(WINDOW_NAME, on_mouse)

    try:
        while True:
            if state["name"] in {"preview", "calibrating"}:
                ret, frame = cap.read()
                if not ret:
                    if state["name"] == "calibrating":
                        error_message = "The video ended before calibration was completed."
                        state["name"] = "failed"
                    else:
                        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue

                results = model(frame, imgsz=480, device=device, conf=person_conf, verbose=False)
                result = results[0]
                assessment = first_person_assessment(result)
                last_detection_ok = bool(
                    assessment
                    and assessment["head_forward_ratio"] is not None
                    and assessment["trunk_angle"] is not None
                )
                last_annotated = result.plot()

                if state["name"] == "calibrating":
                    if assessment is None:
                        collector.total_frames += 1
                    else:
                        collector.add(assessment)
                    frames_left -= 1
                    if frames_left <= 0:
                        try:
                            baseline = collector.finish(
                                min_valid_frames=max(10, int(calibration_frames * 0.25)),
                                min_valid_ratio=0.5,
                            )
                            state["name"] = "review"
                        except CalibrationError as exc:
                            error_message = str(exc)
                            state["name"] = "failed"

            if last_annotated is None:
                continue

            display = fit_for_display(last_annotated.copy())
            height, width = display.shape[:2]
            state["buttons"] = {}

            if state["name"] == "preview":
                detection_text = "Pose detection: OK" if last_detection_ok else "Pose detection: NOT READY"
                detection_color = (90, 230, 90) if last_detection_ok else (80, 80, 255)
                draw_panel(
                    display,
                    [
                        "Sit in your good posture and make sure ear, shoulder, and hip are visible.",
                        detection_text,
                        "Click START when ready.  Space: start / Esc: cancel",
                    ],
                    detection_color,
                )
                button = (max(20, width - 300), height - 82, width - 20, height - 20)
                draw_button(display, "START CALIBRATION", button, last_detection_ok)
                if last_detection_ok:
                    state["buttons"]["start"] = button

            elif state["name"] == "countdown":
                seconds_left = max(0, int(countdown_deadline - time.monotonic()) + 1)
                draw_panel(display, ["Keep your good posture", f"Starting in {seconds_left}..."])
                if time.monotonic() >= countdown_deadline:
                    collector = CalibrationCollector()
                    frames_left = calibration_frames
                    state["name"] = "calibrating"

            elif state["name"] == "calibrating":
                completed = calibration_frames - frames_left
                progress = min(100.0, 100.0 * completed / calibration_frames)
                draw_panel(
                    display,
                    [
                        "CALIBRATING - keep your posture still",
                        f"Progress: {progress:5.1f}%",
                        f"Valid frames: {collector.valid_frames}/{collector.total_frames}",
                    ],
                    (80, 220, 255),
                )

            elif state["name"] == "review":
                draw_panel(
                    display,
                    [
                        "Calibration complete",
                        f"Head baseline: {baseline.head_forward_ratio:.4f}  (MAD {baseline.head_mad:.4f})",
                        f"Trunk baseline: {baseline.trunk_angle:.2f} deg  (MAD {baseline.trunk_mad:.2f})",
                        f"Valid frames: {baseline.valid_frames}/{baseline.total_frames}",
                    ],
                    (90, 230, 90),
                )
                retry = (20, height - 82, 220, height - 20)
                accept = (max(240, width - 300), height - 82, width - 20, height - 20)
                draw_button(display, "RETRY", retry)
                draw_button(display, "ACCEPT", accept)
                state["buttons"] = {"retry": retry, "accept": accept}

            elif state["name"] == "failed":
                draw_panel(
                    display,
                    [
                        "Calibration failed",
                        error_message,
                        "Adjust your position and try again.",
                    ],
                    (80, 80, 255),
                )
                retry = (20, height - 82, 220, height - 20)
                draw_button(display, "RETRY", retry)
                state["buttons"] = {"retry": retry}

            cv2.imshow(WINDOW_NAME, display)
            key = cv2.waitKey(1) & 0xFF
            if cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1:
                return None, model
            action = state.pop("action", None)
            state["action"] = None

            if key == 27:
                return None, model
            if (key == ord(" ") or action == "start") and state["name"] == "preview" and last_detection_ok:
                countdown_deadline = time.monotonic() + countdown_seconds
                state["name"] = "countdown"
            elif (key in (10, 13) or action == "accept") and state["name"] == "review":
                return baseline, model
            elif (key in (ord("r"), ord("R")) or action == "retry") and state["name"] in {"review", "failed"}:
                baseline = None
                error_message = ""
                collector = CalibrationCollector()
                state["name"] = "preview"
    finally:
        cap.release()
        try:
            cv2.destroyWindow(WINDOW_NAME)
        except cv2.error:
            pass


class CalibrationLauncher:
    def __init__(self, root):
        self.root = root
        root.title("Posture assessment")
        root.resizable(False, False)

        project_dir = Path(__file__).resolve().parent
        self.video_var = tk.StringVar()
        self.output_var = tk.StringVar(value=str(project_dir / "output_yolov8"))
        self.model_var = tk.StringVar(value=str(project_dir / "yolov8x-pose.pt"))
        self.duration_var = tk.StringVar(value="5.0")
        self.countdown_var = tk.StringVar(value="3.0")
        self.conf_var = tk.StringVar(value="0.8")
        self.use_calibration_var = tk.BooleanVar(value=True)
        self.status_var = tk.StringVar(value="Select a video to begin.")

        self._build()

    def _build(self):
        frame = ttk.Frame(self.root, padding=18)
        frame.grid(sticky="nsew")

        ttk.Label(frame, text="Posture assessment", font=("Segoe UI", 16, "bold")).grid(
            row=0, column=0, columnspan=3, sticky="w", pady=(0, 14)
        )
        self._path_row(frame, 1, "Input video", self.video_var, self.choose_video)
        self._path_row(frame, 2, "Output folder", self.output_var, self.choose_output)
        self._path_row(frame, 3, "YOLO model", self.model_var, self.choose_model)

        ttk.Label(frame, text="Calibration duration (sec)").grid(row=4, column=0, sticky="w", pady=5)
        ttk.Entry(frame, textvariable=self.duration_var, width=12).grid(row=4, column=1, sticky="w", pady=5)
        ttk.Label(frame, text="Countdown (sec)").grid(row=5, column=0, sticky="w", pady=5)
        ttk.Entry(frame, textvariable=self.countdown_var, width=12).grid(row=5, column=1, sticky="w", pady=5)
        ttk.Label(frame, text="Person confidence").grid(row=6, column=0, sticky="w", pady=5)
        ttk.Entry(frame, textvariable=self.conf_var, width=12).grid(row=6, column=1, sticky="w", pady=5)

        ttk.Checkbutton(
            frame,
            text="Use personal calibration",
            variable=self.use_calibration_var,
        ).grid(row=7, column=0, columnspan=3, sticky="w", pady=(10, 5))

        ttk.Separator(frame).grid(row=8, column=0, columnspan=3, sticky="ew", pady=10)
        ttk.Label(frame, textvariable=self.status_var, wraplength=560).grid(
            row=9, column=0, columnspan=3, sticky="w", pady=(0, 10)
        )
        self.start_button = ttk.Button(frame, text="Start", command=self.start)
        self.start_button.grid(row=10, column=0, columnspan=3, sticky="ew")

    def _path_row(self, parent, row, label, variable, command):
        ttk.Label(parent, text=label).grid(row=row, column=0, sticky="w", pady=5)
        ttk.Entry(parent, textvariable=variable, width=58).grid(row=row, column=1, sticky="ew", padx=8, pady=5)
        ttk.Button(parent, text="Browse", command=command).grid(row=row, column=2, pady=5)

    def choose_video(self):
        path = filedialog.askopenfilename(filetypes=VIDEO_TYPES)
        if not path:
            return
        self.video_var.set(path)

    def choose_output(self):
        path = filedialog.askdirectory(initialdir=self.output_var.get() or None)
        if path:
            self.output_var.set(path)

    def choose_model(self):
        path = filedialog.askopenfilename(filetypes=[("PyTorch weights", "*.pt"), ("All files", "*.*")])
        if path:
            self.model_var.set(path)

    def start(self):
        try:
            video_path = Path(self.video_var.get())
            output_root_text = self.output_var.get().strip()
            model_path = Path(self.model_var.get())
            duration = float(self.duration_var.get())
            countdown = float(self.countdown_var.get())
            person_conf = float(self.conf_var.get())
            if not video_path.is_file():
                raise ValueError("Select an existing input video.")
            if not model_path.is_file():
                raise ValueError("Select an existing YOLO model file.")
            if not output_root_text:
                raise ValueError("Select an output folder.")
            output_root = Path(output_root_text)
            if duration <= 0 or countdown < 0:
                raise ValueError("Duration must be positive and countdown must not be negative.")
            if not 0 < person_conf <= 1:
                raise ValueError("Person confidence must be between 0 and 1.")
        except ValueError as exc:
            messagebox.showerror("Invalid settings", str(exc))
            return

        output_path, csv_path = create_dated_output_paths(video_path, output_root)
        self.start_button.state(["disabled"])
        self.root.update_idletasks()

        try:
            baseline = None
            loaded_model = None
            if self.use_calibration_var.get():
                self.status_var.set("Loading the model and opening calibration preview...")
                self.root.update_idletasks()
                self.root.withdraw()
                baseline, loaded_model = run_calibration_session(
                    str(video_path),
                    str(model_path),
                    duration_seconds=duration,
                    countdown_seconds=countdown,
                    person_conf=person_conf,
                )
                self.root.deiconify()
                if baseline is None:
                    self.status_var.set("Calibration was cancelled.")
                    return

            self.status_var.set("Processing the video. Progress is shown in the terminal.")
            self.root.update_idletasks()
            process_video_yolov8(
                str(video_path),
                output_path=str(output_path),
                csv_path=str(csv_path),
                model_name=str(model_path),
                person_conf=person_conf,
                calibration_baseline=baseline,
                model=loaded_model,
            )
            self.status_var.set(f"Completed: {output_path}")
            messagebox.showinfo("Completed", f"Processing completed.\n\n{output_path}")
        except Exception as exc:
            self.root.deiconify()
            self.status_var.set("Processing failed.")
            messagebox.showerror("Error", str(exc))
        finally:
            self.start_button.state(["!disabled"])


def main():
    os.chdir(Path(__file__).resolve().parent)
    root = tk.Tk()
    CalibrationLauncher(root)
    root.mainloop()


if __name__ == "__main__":
    main()
