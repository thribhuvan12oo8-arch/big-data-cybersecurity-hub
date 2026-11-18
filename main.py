"""Run privacy-preserving, rule-based CCTV behaviour monitoring."""

from __future__ import annotations

import argparse
import math
import time
from collections import deque
from dataclasses import replace
from datetime import datetime
from pathlib import Path

import cv2

from src.alert_engine import AlertDecision, AlertLogger, classify_behaviour
from src.anonymizer import PrivacyAnonymizer
from src.camera_motion import CameraMotionGuard
from src.detector import PersonDetector
from src.face_detector import FaceDetector
from src.fall_detector import FallDetector, FallObservation
from src.metrics import RunMetrics
from src.monitoring_panel import MonitoringPanel, PersonPanelData
from src.motion_analyzer import MotionAnalyzer
from src.line_crossing_monitor import LineCrossingMonitor
from src.observation_logger import ObservationLogger
from src.pose_estimator import (
    PoseEstimator,
    associate_poses_to_tracks,
    draw_skeletons,
)
from src.tracker import PersonTracker
from src.utils import ensure_output_directories, load_config
from src.zone_monitor import ZoneMonitor
from evaluation.diagnostics import FrameDiagnosticWriter
from evaluation.environment import ProcessResourceSampler, environment_snapshot


def _parse_source(source: str | int) -> str | int:
    if isinstance(source, int):
        return source
    return int(source) if source.isdecimal() else source


def _intersection_over_union(
    first: tuple[int, int, int, int], second: tuple[int, int, int, int]
) -> float:
    left = max(first[0], second[0])
    top = max(first[1], second[1])
    right = min(first[2], second[2])
    bottom = min(first[3], second[3])
    intersection = max(0, right - left) * max(0, bottom - top)
    if intersection == 0:
        return 0.0
    first_area = max(0, first[2] - first[0]) * max(0, first[3] - first[1])
    second_area = max(0, second[2] - second[0]) * max(0, second[3] - second[1])
    return intersection / max(1, first_area + second_area - intersection)


def select_source() -> str | int | None:
    """Display a small Tk source selector and return the selected input."""
    try:
        from tkinter import Tk, filedialog, ttk

        root = Tk()
    except ImportError as exc:
        raise RuntimeError(
            "Tkinter is unavailable; pass --source with a camera index or video path."
        ) from exc
    except Exception as exc:
        raise RuntimeError(f"Could not open the source selector: {exc}") from exc

    selected_source: str | int | None = None
    root.title("Choose CCTV Source")
    root.resizable(False, False)
    root.geometry("380x170")

    container = ttk.Frame(root, padding=18)
    container.pack(fill="both", expand=True)
    ttk.Label(
        container,
        text="Select a video source",
        font=("Segoe UI", 13, "bold"),
    ).pack(pady=(0, 14))

    def start_camera() -> None:
        nonlocal selected_source
        selected_source = 0
        root.destroy()

    def select_recording() -> None:
        nonlocal selected_source
        video_path = filedialog.askopenfilename(
            parent=root,
            title="Select a recorded video",
            filetypes=[
                (
                    "Video files",
                    "*.mp4 *.avi *.mov *.mkv *.m4v *.wmv",
                ),
                ("All files", "*.*"),
            ],
        )
        if video_path:
            selected_source = video_path
        root.destroy()

    buttons = ttk.Frame(container)
    buttons.pack(fill="x")
    ttk.Button(
        buttons, text="START LIVE CAMERA", command=start_camera
    ).pack(fill="x", pady=4)
    ttk.Button(
        buttons, text="SELECT RECORDED VIDEO", command=select_recording
    ).pack(fill="x", pady=4)
    root.protocol("WM_DELETE_WINDOW", root.destroy)
    try:
        root.mainloop()
    except Exception as exc:
        raise RuntimeError(f"Source selection failed: {exc}") from exc
    return selected_source


def _create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Privacy-preserving, rule-based CCTV behaviour analysis."
    )
    parser.add_argument(
        "--source",
        default=None,
        help="Video path or webcam index; if omitted, show the source selector.",
    )
    parser.add_argument("--config", default="config.yaml", help="YAML configuration.")
    parser.add_argument("--model", help="Override the configured YOLOv8 weights path.")
    parser.add_argument(
        "--device",
        default="auto",
        help="Inference device (auto, cpu, 0, etc.; default: auto).",
    )
    parser.add_argument(
        "--save",
        action="store_true",
        help="Save the masked CCTV footage at the source frame size.",
    )
    parser.add_argument(
        "--no-display",
        action="store_true",
        help="Process without opening the OpenCV monitoring window.",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        help="Optional maximum frame count, useful for bounded test runs.",
    )
    parser.add_argument("--debug", action="store_true", help="Show full exceptions.")
    parser.add_argument(
        "--profile-deepsort-internals",
        action="store_true",
        help="Record per-stage DeepSORT timings in the run metrics JSON.",
    )
    parser.add_argument(
        "--evaluation-export",
        help=(
            "Optional local JSONL path for per-frame detections, tracks, face/mask "
            "boxes, rule alerts, timestamps, and stage timings."
        ),
    )
    parser.add_argument(
        "--evaluation-video-id",
        help="Stable annotation video ID for --evaluation-export (defaults to filename).",
    )
    return parser


def run(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    ensure_output_directories(config)
    if args.source is None:
        raise ValueError("A video source must be selected before processing.")
    source = _parse_source(args.source)
    is_live_camera = isinstance(source, int)
    source_label = "LIVE CAMERA" if is_live_camera else Path(source).name
    source_log_name = f"camera:{source}" if is_live_camera else str(source)
    capture = cv2.VideoCapture(source)
    if not capture.isOpened():
        capture.release()
        raise ValueError(
            f"Could not open video source '{args.source}'. Check the file path, "
            "webcam index, and camera permissions."
        )

    fps = float(capture.get(cv2.CAP_PROP_FPS))
    source_fps_origin = "capture_metadata"
    if not math.isfinite(fps) or fps <= 0:
        fps = 30.0
        source_fps_origin = "fallback_30_fps"
    frame_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if frame_width <= 0 or frame_height <= 0:
        capture.release()
        raise ValueError("The selected source did not report a valid frame size.")

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    alert_path = Path(config["output"]["logs_dir"]) / f"alerts_{run_id}.csv"
    metrics = RunMetrics(source_log_name, source_fps=fps)
    metrics.source_fps_origin = source_fps_origin
    metrics.source_timing_mode = (
        "monotonic_clock" if is_live_camera else "pending_media_timestamp_check"
    )
    evaluation_export_path = getattr(args, "evaluation_export", None)
    evaluation_video_id = getattr(args, "evaluation_video_id", None) or (
        f"camera:{source}" if is_live_camera else Path(source).name
    )
    if evaluation_export_path:
        source_log_name = evaluation_video_id
        metrics.source = evaluation_video_id
    diagnostic_writer: FrameDiagnosticWriter | None = None
    resource_sampler: ProcessResourceSampler | None = None
    process_every = max(1, int(config["performance"]["process_every_n_frames"]))
    pose_config = config.get("pose") or {}
    pose_enabled = bool(pose_config.get("enabled", False))
    display = bool(config["display"]["show_video"]) and not args.no_display
    metrics.run_options = {
        "source_resolution": [frame_width, frame_height],
        "source_fps": fps if source_fps_origin == "capture_metadata" else None,
        "configured_yolo_model": args.model or config["detection"]["model"],
        "configured_inference_size": int(
            config["performance"]["inference_size"]
        ),
        "process_every_n_frames": process_every,
        "display_enabled": display,
        "video_saving_enabled": bool(args.save),
        "max_frames": args.max_frames,
        "requested_device": args.device,
        "cpu_torch_threads": int(config["performance"]["cpu_torch_threads"]),
        "face_detection_max_width": int(
            config["privacy"].get("face_detection_max_width", 480)
        ),
        "deepsort_internal_profiling": bool(
            getattr(args, "profile_deepsort_internals", False)
        ),
        "evaluation_export_enabled": bool(evaluation_export_path),
        "evaluation_video_id": evaluation_video_id if evaluation_export_path else None,
        "line_crossing_enabled": bool(config["line_crossing"]["enabled"]),
        "pose_enabled": pose_enabled,
        "pose_model": (
            str(pose_config.get("model", "yolov8n-pose.pt"))
            if pose_enabled
            else None
        ),
        "pose_inference_interval_frames": process_every if pose_enabled else None,
        "pose_detection_confidence": (
            float(pose_config.get("confidence", 0.25)) if pose_enabled else None
        ),
        "pose_keypoint_confidence": (
            float(pose_config.get("keypoint_confidence", 0.35))
            if pose_enabled
            else None
        ),
        "pose_association_iou": (
            float(pose_config.get("association_iou", 0.30))
            if pose_enabled
            else None
        ),
    }
    writer: cv2.VideoWriter | None = None
    initialization_started_ns = time.perf_counter_ns()
    try:
        detector = PersonDetector(
            model_path=args.model or config["detection"]["model"],
            confidence=float(config["detection"]["confidence"]),
            image_size=int(config["performance"]["inference_size"]),
            device=args.device,
            nms_iou=float(config["detection"].get("nms_iou", 0.7)),
            use_half_precision_on_cuda=bool(
                config["performance"]["use_half_precision_on_cuda"]
            ),
        )
        if detector.device == "cpu":
            import torch

            torch.set_num_threads(
                max(1, int(config["performance"]["cpu_torch_threads"]))
            )
        tracker = PersonTracker(
            max_age=int(config["tracking"]["max_age"]),
            n_init=int(config["tracking"]["n_init"]),
            max_cosine_distance=float(config["tracking"]["max_cosine_distance"]),
            max_iou_distance=float(config["tracking"]["max_iou_distance"]),
            nn_budget=int(config["tracking"]["nn_budget"]),
            device=detector.device,
            profile_internals=bool(
                getattr(args, "profile_deepsort_internals", False)
            ),
        )
        anonymizer = PrivacyAnonymizer(
            pixel_block_size=int(config["privacy"]["pixel_block_size"]),
            blur_kernel_size=int(config["privacy"]["blur_kernel_size"]),
        )
        face_detector = FaceDetector(
            maximum_width=int(config["privacy"].get("face_detection_max_width", 480))
        )
        movement_analyzer = MotionAnalyzer(**config["movement"])
        zone_monitor = ZoneMonitor(
            bounds=config["zone"]["bounds"],
            stale_track_seconds=float(config["movement"]["stale_track_seconds"]),
            hysteresis_fraction=float(config["zone"]["hysteresis_fraction"]),
        )
        line_crossing_monitor = LineCrossingMonitor(
            **config["line_crossing"]
        )
        fall_detector = FallDetector(
            **config["fall"],
            stale_track_seconds=float(config["movement"]["stale_track_seconds"]),
        )
        alert_logger = AlertLogger(
            alert_path, cooldown_seconds=float(config["alerts"]["cooldown_seconds"])
        )
        panel = MonitoringPanel(panel_width=int(config["display"]["panel_width"]))
        camera_motion_guard = CameraMotionGuard(**config["camera_motion"])
        pose_estimator = (
            PoseEstimator(
                model_path=pose_config.get("model", "yolov8n-pose.pt"),
                detection_confidence=float(pose_config.get("confidence", 0.25)),
                image_size=int(config["performance"]["inference_size"]),
                device=detector.device,
            )
            if pose_enabled
            else None
        )
        metrics.record_initialization(
            (time.perf_counter_ns() - initialization_started_ns) / 1_000_000_000
        )
        metrics.run_options["actual_device"] = getattr(
            detector, "device", args.device
        )
        metrics.run_options["model_weights"] = str(
            getattr(
                getattr(detector, "model", None),
                "ckpt_path",
                args.model or config["detection"]["model"],
            )
        )
        metrics.runtime_environment = environment_snapshot(detector.device)
    except Exception:
        capture.release()
        raise
    zone_bounds = zone_monitor.pixel_bounds(frame_width, frame_height)

    if args.save:
        output_path = (
            Path(config["output"]["processed_video_dir"])
            / f"processed_{run_id}.mp4"
        )
        writer = cv2.VideoWriter(
            str(output_path),
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps,
            (frame_width, frame_height),
        )
        if not writer.isOpened():
            writer.release()
            capture.release()
            raise RuntimeError(f"Could not create processed video: {output_path}")
        print(f"Processed video: {output_path}")

    detection_log_path = Path(config["output"]["logs_dir"]) / (
        f"detections_{run_id}.csv"
    )
    tracking_log_path = Path(config["output"]["logs_dir"]) / f"tracks_{run_id}.csv"
    diagnostics_log_path = (
        Path(config["output"]["logs_dir"]) / f"tracker_diagnostics_{run_id}.csv"
    )
    try:
        observation_logger = ObservationLogger(
            detection_log_path, tracking_log_path, diagnostics_log_path
        )
        if evaluation_export_path:
            diagnostic_writer = FrameDiagnosticWriter(
                evaluation_export_path, evaluation_video_id
            )
            resource_sampler = ProcessResourceSampler()
    except Exception:
        if writer is not None:
            writer.release()
        capture.release()
        if diagnostic_writer is not None:
            diagnostic_writer.close()
        raise

    fps_description = (
        f"{fps:.2f} FPS"
        if source_fps_origin == "capture_metadata"
        else "FPS unavailable; using a 30 FPS fallback"
    )
    print(f"Source: {source_label} ({frame_width}x{frame_height}, {fps_description})")
    print(f"Alerts CSV: {alert_path}")
    frame_number = 0
    paused = False
    target_display_fps = max(1.0, float(config["performance"]["target_display_fps"]))
    presentation_interval = 1.0 / fps
    presentation_stride = max(1, math.ceil(fps / target_display_fps))
    max_display_stale_frames = process_every - 1
    last_panel_data: dict[str, PersonPanelData] = {}
    last_line_crossings: dict[str, tuple[str, float]] = {}
    run_clock_started = time.monotonic()
    previous_video_timestamp: float | None = None
    use_media_timestamps = not is_live_camera
    display_timestamps: deque[float] = deque(maxlen=30)
    camera_motion_active = False
    metrics.start_processing()
    frame_stage_timings: dict[str, float] | None = None

    def record_stage(
        name: str, elapsed_seconds: float, warmup: bool = False
    ) -> None:
        metrics.record_stage(name, elapsed_seconds, warmup=warmup)
        if frame_stage_timings is not None:
            frame_stage_timings[name] = (
                frame_stage_timings.get(name, 0.0) + max(0.0, elapsed_seconds)
            )

    try:
        while args.max_frames is None or frame_number < args.max_frames:
            if display and not is_live_camera and paused:
                key = cv2.waitKey(30) & 0xFF
                if key in (ord("q"), 27):
                    break
                if key == ord(" "):
                    paused = False
                    print("Video resumed.")
                continue

            frame_loop_started_ns = time.perf_counter_ns()
            frame_stage_timings = {}
            frame_events: list[dict[str, object]] = []
            decode_started_ns = frame_loop_started_ns
            ok, frame = capture.read()
            decode_elapsed = (
                time.perf_counter_ns() - decode_started_ns
            ) / 1_000_000_000
            if ok and frame is not None:
                record_stage(
                    "capture_read",
                    decode_elapsed,
                    warmup=frame_number == 0,
                )
            if not ok or frame is None:
                break
            frame_number += 1
            warmup_frame = frame_number == metrics.warmup_frame_number
            if is_live_camera:
                video_timestamp = time.monotonic() - run_clock_started
            else:
                reported_timestamp = (
                    float(capture.get(cv2.CAP_PROP_POS_MSEC)) / 1000.0
                )
                if (
                    use_media_timestamps
                    and math.isfinite(reported_timestamp)
                    and reported_timestamp >= 0
                    and (
                        previous_video_timestamp is None
                        or reported_timestamp > previous_video_timestamp
                    )
                ):
                    video_timestamp = reported_timestamp
                    metrics.source_timing_mode = "opencv_media_timestamps"
                else:
                    use_media_timestamps = False
                    step = 1.0 / fps
                    video_timestamp = (
                        previous_video_timestamp + step
                        if previous_video_timestamp is not None
                        else 0.0
                    )
                    metrics.source_timing_mode = "frame_rate_fallback"
                previous_video_timestamp = video_timestamp
            frame_camera_motion_started_ns = time.perf_counter_ns()
            camera_observation = camera_motion_guard.update(frame)
            record_stage(
                "camera_motion",
                (time.perf_counter_ns() - frame_camera_motion_started_ns)
                / 1_000_000_000,
                warmup=warmup_frame,
            )
            camera_motion_active = camera_observation.moving
            metrics.record_camera_motion(camera_motion_active)
            movement_analyzer.advance_frame(camera_motion_active)

            should_detect = (frame_number - 1) % process_every == 0
            detections = []
            if should_detect:
                detection_started_ns = time.perf_counter_ns()
                detections = detector.detect(frame)
                record_stage(
                    "yolo_inference",
                    (time.perf_counter_ns() - detection_started_ns)
                    / 1_000_000_000,
                    warmup=warmup_frame,
                )
                metrics.record_detections(len(detections))
                inference_shape = getattr(detector, "last_inference_shape", None)
                metrics.run_options["actual_inference_resolution"] = (
                    list(inference_shape)
                    if inference_shape is not None
                    else None
                )
            detection_logging_started_ns = time.perf_counter_ns()
            observation_logger.record_detections(
                source_log_name, frame_number, detections
            )
            record_stage(
                "detection_logging",
                (time.perf_counter_ns() - detection_logging_started_ns)
                / 1_000_000_000,
                warmup=warmup_frame,
            )
            tracking_started_ns = time.perf_counter_ns()
            tracks = tracker.update(
                detections,
                frame,
                max_stale_frames=max_display_stale_frames,
                privacy_max_stale_frames=int(
                    config["tracking"]["privacy_max_stale_frames"]
                ),
                inference_performed=should_detect,
            )
            record_stage(
                "deepsort_update",
                (time.perf_counter_ns() - tracking_started_ns) / 1_000_000_000,
                warmup=warmup_frame,
            )
            metrics.record_tracks(tracker.last_diagnostics)
            poses_by_track = {}
            if pose_estimator is not None and should_detect and any(
                person.observed and person.confirmed for person in tracks
            ):
                pose_started_ns = time.perf_counter_ns()
                pose_detections = pose_estimator.estimate(frame)
                poses_by_track = associate_poses_to_tracks(
                    pose_detections,
                    tracks,
                    minimum_iou=float(pose_config.get("association_iou", 0.30)),
                )
                record_stage(
                    "pose_inference",
                    (time.perf_counter_ns() - pose_started_ns) / 1_000_000_000,
                    warmup=warmup_frame,
                )
            line_crossings: dict[str, tuple[str, float]] = {}
            line_crossing_started_ns = time.perf_counter_ns()
            line_crossing_monitor.cleanup(video_timestamp)
            for person in tracks:
                if not person.observed or not person.confirmed:
                    continue
                crossing = line_crossing_monitor.update(
                    person, video_timestamp, frame_width, frame_height
                )
                if crossing.crossed and crossing.direction is not None:
                    crossing_timestamp = crossing.source_timestamp_seconds
                    if crossing_timestamp is None:
                        continue
                    line_crossings[person.track_id] = (
                        crossing.direction,
                        crossing_timestamp,
                    )
                    last_line_crossings[person.track_id] = (
                        crossing.direction,
                        crossing_timestamp,
                    )
            record_stage(
                "line_crossing_monitor",
                (time.perf_counter_ns() - line_crossing_started_ns)
                / 1_000_000_000,
                warmup=warmup_frame,
            )
            active_tracks = [
                person
                for person in tracks
                if person.observed
                or person.frames_since_update <= max_display_stale_frames
            ]
            tracking_logging_started_ns = time.perf_counter_ns()
            observation_logger.record_tracks(
                source_log_name, frame_number, active_tracks
            )
            observation_logger.record_tracker_diagnostics(
                source_log_name,
                frame_number,
                should_detect,
                len(detections),
                tracker.last_diagnostics,
            )
            record_stage(
                "observation_logging",
                (time.perf_counter_ns() - tracking_logging_started_ns)
                / 1_000_000_000,
                warmup=warmup_frame,
            )

            face_detection_started_ns = time.perf_counter_ns()
            face_boxes = face_detector.detect(frame)
            record_stage(
                "face_detection",
                (time.perf_counter_ns() - face_detection_started_ns)
                / 1_000_000_000,
                warmup=warmup_frame,
            )
            face_detection_shape = getattr(face_detector, "last_input_shape", None)
            metrics.run_options["face_detection_resolution"] = (
                list(face_detection_shape)
                if face_detection_shape is not None
                else None
            )
            processed = frame.copy()
            observed_boxes = [person.bbox for person in tracks if person.observed]
            fallback_boxes = [person.bbox for person in tracks]
            for detection in detections:
                if not any(
                    _intersection_over_union(detection.bbox, tracked_box) >= 0.5
                    for tracked_box in observed_boxes
                ):
                    fallback_boxes.append(detection.bbox)
            anonymization_started_ns = time.perf_counter_ns()
            privacy_summary = anonymizer.anonymize_frame(
                processed, face_boxes, fallback_boxes
            )
            record_stage(
                "privacy_masking",
                (time.perf_counter_ns() - anonymization_started_ns)
                / 1_000_000_000,
                warmup=warmup_frame,
            )
            metrics.record_privacy(
                privacy_summary.detected_faces,
                privacy_summary.masked_faces,
                privacy_summary.fallback_regions,
            )
            if poses_by_track:
                pose_render_started_ns = time.perf_counter_ns()
                draw_skeletons(
                    processed,
                    poses_by_track,
                    keypoint_confidence=float(
                        pose_config.get("keypoint_confidence", 0.35)
                    ),
                )
                record_stage(
                    "pose_visualization",
                    (time.perf_counter_ns() - pose_render_started_ns)
                    / 1_000_000_000,
                    warmup=warmup_frame,
                )
            privacy_mapping_started_ns = time.perf_counter_ns()
            privacy_by_track: dict[str, tuple[str, bool]] = {
                person.track_id: (
                    anonymizer.person_privacy_status(person.bbox, face_boxes),
                    True,
                )
                for person in tracks
            }
            record_stage(
                "privacy_status_mapping",
                (time.perf_counter_ns() - privacy_mapping_started_ns)
                / 1_000_000_000,
                warmup=warmup_frame,
            )
            panel_people: list[PersonPanelData] = []
            behaviour_started_ns = time.perf_counter_ns()
            nested_analysis_seconds = 0.0

            for person in active_tracks:
                privacy_status, privacy_masked = privacy_by_track[person.track_id]
                cached_data = last_panel_data.get(person.track_id)
                if person.observed:
                    movement_started_ns = time.perf_counter_ns()
                    movement = movement_analyzer.update(
                        person,
                        video_timestamp,
                        camera_motion=camera_motion_active,
                    )
                    cached_data = last_panel_data.get(person.track_id)
                    if movement.valid_bbox and (
                        movement.valid_observation or movement.displacement_px == 0
                    ):
                        zone = zone_monitor.update(
                            person,
                            movement,
                            video_timestamp,
                            frame_width,
                            frame_height,
                        )
                        fall = fall_detector.update(person, movement, video_timestamp)
                    elif cached_data is not None:
                        # Rejected jumps/gaps must not become zone or fall evidence.
                        zone = cached_data.zone
                        fall = cached_data.fall
                    else:
                        zone = zone_monitor.update(
                            person,
                            movement,
                            video_timestamp,
                            frame_width,
                            frame_height,
                        )
                        fall = FallObservation(person.track_id, "NO FALL", False)
                    record_stage(
                        "movement_analysis",
                        movement_elapsed := (
                            time.perf_counter_ns() - movement_started_ns
                        )
                        / 1_000_000_000,
                        warmup=warmup_frame,
                    )
                    nested_analysis_seconds += movement_elapsed
                    alert_started_ns = time.perf_counter_ns()
                    decision = classify_behaviour(movement, zone, fall)
                    metrics.record_person(person.confidence, privacy_masked)
                    alert_written = alert_logger.record_if_new(
                        decision=decision,
                        person=person,
                        movement=movement,
                        zone=zone,
                        fall=fall,
                        privacy_status=privacy_status,
                        frame_number=frame_number,
                        video=source_log_name,
                        elapsed_seconds=video_timestamp,
                    )
                    if alert_written:
                        metrics.record_alert(decision.alert_type)
                        frame_events.append(
                            {
                                "event_type": decision.event_type,
                                "track_id": person.track_id,
                                "alert_type": decision.alert_type,
                                "source_timestamp_seconds": video_timestamp,
                            }
                        )
                    crossing_event = line_crossings.get(person.track_id)
                    if crossing_event is not None:
                        direction, crossing_timestamp = crossing_event
                        alert_logger.record_line_crossing(
                            person=person,
                            movement=movement,
                            zone=zone,
                            fall=fall,
                            privacy_status=privacy_status,
                            frame_number=frame_number,
                            video=source_log_name,
                            source_timestamp_seconds=crossing_timestamp,
                            direction=direction,
                        )
                        metrics.record_alert("SECURITY ALERT")
                        frame_events.append(
                            {
                                "event_type": "line_crossing",
                                "track_id": person.track_id,
                                "alert_type": "SECURITY ALERT",
                                "direction": direction,
                                "source_timestamp_seconds": crossing_timestamp,
                            }
                        )
                        if decision.priority < 2:
                            decision = AlertDecision(
                                f"LINE CROSSING {direction.upper()}",
                                "SECURITY ALERT",
                                "line_crossing",
                                2,
                            )
                    record_stage(
                        "alert_processing",
                        alert_elapsed := (
                            time.perf_counter_ns() - alert_started_ns
                        )
                        / 1_000_000_000,
                        warmup=warmup_frame,
                    )
                    nested_analysis_seconds += alert_elapsed
                    cached_data = PersonPanelData(
                        person=person,
                        movement=movement,
                        zone=zone,
                        fall=fall,
                        privacy_status=privacy_status,
                        decision=decision,
                        line_crossing_direction=(
                            last_line_crossings.get(person.track_id, (None, None))[0]
                        ),
                        line_crossing_timestamp_seconds=(
                            last_line_crossings.get(person.track_id, (None, None))[1]
                        ),
                    )
                    last_panel_data[person.track_id] = cached_data
                elif cached_data is not None:
                    cached_data = replace(
                        cached_data, person=person, privacy_status=privacy_status
                    )
                    last_panel_data[person.track_id] = cached_data
                if cached_data is not None:
                    panel_people.append(cached_data)
            record_stage(
                "person_panel_overhead",
                max(
                    0.0,
                    (time.perf_counter_ns() - behaviour_started_ns)
                    / 1_000_000_000
                    - nested_analysis_seconds,
                ),
                warmup=warmup_frame,
            )

            active_ids = {person.track_id for person in active_tracks}
            last_panel_data = {
                track_id: data
                for track_id, data in last_panel_data.items()
                if track_id in active_ids
            }
            last_line_crossings = {
                track_id: crossing
                for track_id, crossing in last_line_crossings.items()
                if track_id in active_ids
            }
            rendering_started_ns = time.perf_counter_ns()
            zone_bounds = zone_monitor.pixel_bounds(frame_width, frame_height)
            panel.draw_zone(processed, zone_bounds)
            panel.draw_line_crossing(
                processed,
                line_crossing_monitor.pixel_endpoints(frame_width, frame_height),
            )
            composed = None
            display_due = display and (
                frame_number == 1 or frame_number % presentation_stride == 0
            )
            if display_due:
                current_fps = metrics.processing_fps()
                average_latency = metrics.average_steady_latency_ms()
                composed = panel.compose(
                    processed,
                    panel_people,
                    current_fps,
                    frame_number,
                    source_label,
                    "RUNNING",
                    source_fps=(
                        fps if source_fps_origin == "capture_metadata" else None
                    ),
                    display_fps=(
                        (len(display_timestamps) - 1)
                        / max(
                            0.001,
                            display_timestamps[-1] - display_timestamps[0],
                        )
                        if len(display_timestamps) > 1
                        else 0.0
                    ),
                    latency_ms=average_latency,
                    camera_motion=camera_motion_active,
                    inference_fps=metrics.inference_fps(),
                    detections=len(detections),
                    confirmed_tracks=tracker.last_diagnostics["active_tracks"],
                    tentative_tracks=tracker.last_diagnostics["tentative_tracks"],
                    lost_tracks=tracker.last_diagnostics["lost_tracks"],
                    active_alerts=sum(
                        data.decision.alert_type != "NO ALERT"
                        for data in panel_people
                    ),
                    line_crossing_enabled=line_crossing_monitor.enabled,
                )
                max_display_width = int(config["display"].get("max_width", 1440))
                if composed.shape[1] > max_display_width:
                    scale = max_display_width / composed.shape[1]
                    composed = cv2.resize(
                        composed,
                        (
                            max_display_width,
                            max(1, round(composed.shape[0] * scale)),
                        ),
                        interpolation=cv2.INTER_AREA,
                    )
            record_stage(
                "monitoring_panel_rendering",
                (time.perf_counter_ns() - rendering_started_ns)
                / 1_000_000_000,
                warmup=warmup_frame,
            )
            if writer is not None:
                writer_started_ns = time.perf_counter_ns()
                writer.write(processed)
                record_stage(
                    "video_writing",
                    (time.perf_counter_ns() - writer_started_ns)
                    / 1_000_000_000,
                    warmup=warmup_frame,
                )
            frame_elapsed = (
                time.perf_counter_ns() - frame_loop_started_ns
            ) / 1_000_000_000
            record_stage(
                "end_to_end_frame",
                frame_elapsed,
                warmup=warmup_frame,
            )
            metrics.record_frame(frame_elapsed, warmup=warmup_frame)
            if resource_sampler is not None:
                cpu_percent, rss_bytes = resource_sampler.sample()
                metrics.record_resource_sample(
                    cpu_percent, rss_bytes, warmup=warmup_frame
                )
            if diagnostic_writer is not None:
                diagnostic_writer.record(
                    frame_number=frame_number,
                    source_timestamp_seconds=video_timestamp,
                    frame_width=frame_width,
                    frame_height=frame_height,
                    detections=detections,
                    tracks=tracks,
                    tracker_state=tracker.last_diagnostics,
                    face_boxes=face_boxes,
                    anonymization=privacy_summary,
                    events=frame_events,
                    stage_timings_seconds=frame_stage_timings or {},
                )

            if display:
                remaining_ms = max(
                    1,
                    int(
                        1000
                        * (
                            presentation_interval
                            - (
                                time.perf_counter_ns() - frame_loop_started_ns
                            )
                            / 1_000_000_000
                        )
                    ),
                )
                display_started_ns = time.perf_counter_ns()
                if display_due and composed is not None:
                    cv2.imshow(config["display"]["window_name"], composed)
                    display_timestamps.append(time.perf_counter())
                    metrics.record_presented_frame()
                key = cv2.waitKey(remaining_ms) & 0xFF
                record_stage(
                    "display_and_key_poll",
                    (time.perf_counter_ns() - display_started_ns)
                    / 1_000_000_000,
                    warmup=warmup_frame,
                )
                if key in (ord("q"), 27):
                    break
                if not is_live_camera and key == ord(" "):
                    paused = True
                    print("Video paused. Press SPACE to resume.")
        metrics.finalization_status = "completed"
    except KeyboardInterrupt:
        metrics.finalization_status = "interrupted"
        raise
    except Exception:
        metrics.finalization_status = "failed"
        raise
    finally:
        metrics.finish_processing()
        capture.release()
        if writer is not None:
            writer.release()
        observation_logger.close()
        if diagnostic_writer is not None:
            diagnostic_writer.close()
        if display:
            cv2.destroyAllWindows()
        profile_snapshot = getattr(tracker, "profile_snapshot", None)
        if callable(profile_snapshot):
            metrics.deepsort_internal_profile = profile_snapshot()
        metrics_path = metrics.save(config["output"]["metrics_dir"])

    print(f"Frames processed: {metrics.frames_processed}")
    print(f"Average processing FPS: {metrics.snapshot()['average_fps']}")
    print(f"Run metrics: {metrics_path}")
    print(f"Alert log: {alert_path}")
    print(f"Detection annotations: {detection_log_path}")
    print(f"Tracking annotations: {tracking_log_path}")
    print(f"Tracker diagnostics: {diagnostics_log_path}")
    if args.debug:
        snapshot = metrics.snapshot()
        print("Average stage latency (ms):")
        for stage, seconds in snapshot["stage_timing_seconds"].items():
            calls = snapshot["stage_timing_calls"][stage]
            print(f"  {stage}: {1000.0 * seconds / max(1, calls):.2f}")
    return 0


def main() -> int:
    parser = _create_parser()
    args = parser.parse_args()
    if args.max_frames is not None and args.max_frames <= 0:
        parser.error("--max-frames must be a positive integer.")
    try:
        if args.source is None:
            args.source = select_source()
            if args.source is None:
                print("No video selected.")
                return 0
        return run(args)
    except KeyboardInterrupt:
        print("Processing interrupted.")
        return 130
    except (ValueError, RuntimeError, OSError) as exc:
        if args.debug:
            raise
        parser.exit(2, f"Error: {exc}\n")


if __name__ == "__main__":
    raise SystemExit(main())
