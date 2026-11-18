from __future__ import annotations

import csv
import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from evaluation.annotations import validate_annotations
from evaluation.annotate import HEADERS, _open_annotation_output, annotate_video
from evaluation.diagnostics import export_from_csv_logs
from evaluation.evaluate import (
    evaluate_detections,
    evaluate_events,
    evaluate_performance,
    evaluate_privacy,
    evaluate_tracking,
    iou,
    main as evaluate_main,
)
from evaluation.motchallenge import (
    _convert_prediction_csv,
    _copy_motchallenge_tracking,
)
from evaluation.sample_frames import sample_frames
from evaluation.report import generate_report
from evaluation.templates import create_templates
from src.metrics import RunMetrics


def _write(path, header: str, rows: list[str]) -> None:
    path.write_text(header + "\n" + "\n".join(rows) + "\n", encoding="utf-8")


def test_iou_calculation_and_empty_intersection() -> None:
    assert iou((0, 0, 10, 10), (5, 0, 15, 10)) == pytest.approx(1 / 3)
    assert iou((0, 0, 10, 10), (10, 10, 15, 15)) == 0


def test_detection_matching_handles_duplicate_predictions_fp_and_fn(tmp_path) -> None:
    truth = tmp_path / "truth.csv"
    predictions = tmp_path / "predictions.csv"
    _write(
        truth,
        "video,frame_number,person_id,x1,y1,x2,y2\n",
        ["clip,1,,0,0,10,10", "clip,1,,20,0,30,10"],
    )
    _write(
        predictions,
        "video,frame_number,person_id,x1,y1,x2,y2,confidence,class_id\n",
        [
            "clip,1,,0,0,10,10,0.9,0",
            "clip,1,,0,0,10,10,0.8,0",
            "clip,2,,50,50,60,60,0.7,0",
        ],
    )
    result = evaluate_detections(truth, predictions)
    assert (result["TP"], result["FP"], result["FN"]) == (1, 2, 1)
    assert result["precision"] == pytest.approx(1 / 3)
    assert result["recall"] == pytest.approx(1 / 2)


def test_detection_average_precision_uses_confidence_and_coco_thresholds(tmp_path) -> None:
    truth = tmp_path / "truth.csv"
    predictions = tmp_path / "predictions.csv"
    _write(
        truth,
        "video,frame_number,person_id,x1,y1,x2,y2\n",
        ["clip,1,,0,0,10,10"],
    )
    _write(
        predictions,
        "video,frame_number,person_id,x1,y1,x2,y2,confidence,class_id\n",
        ["clip,1,,0,0,10,10,0.95,0"],
    )
    result = evaluate_detections(truth, predictions)
    assert result["AP50"] == pytest.approx(1)
    assert result["mAP50_95"] == pytest.approx(1)


def test_detection_ap_is_undefined_without_ground_truth(tmp_path) -> None:
    truth = tmp_path / "truth.csv"
    predictions = tmp_path / "predictions.csv"
    _write(truth, "video,frame_number,person_id,x1,y1,x2,y2\n", [])
    _write(
        predictions,
        "video,frame_number,person_id,x1,y1,x2,y2,confidence\n",
        ["clip,1,,0,0,10,10,0.8"],
    )
    assert evaluate_detections(truth, predictions)["AP50"] is None


def test_tracking_identity_switch_and_fragmentation(tmp_path) -> None:
    truth = tmp_path / "truth.csv"
    predictions = tmp_path / "tracks.csv"
    _write(
        truth,
        "video,frame_number,person_id,x1,y1,x2,y2\n",
        [f"clip,{frame},A,0,0,10,10" for frame in range(1, 6)],
    )
    _write(
        predictions,
        "video,frame_number,person_id,x1,y1,x2,y2\n",
        [
            "clip,1,ID1,0,0,10,10",
            "clip,2,ID1,0,0,10,10",
            "clip,4,ID1,0,0,10,10",
            "clip,5,ID2,0,0,10,10",
        ],
    )
    result = evaluate_tracking(truth, predictions)
    assert result["ID_switches"] == 1
    assert result["fragmentations"] == 1
    assert result["FN"] == 1
    assert result["HOTA"] is None


def test_event_matching_is_one_to_one_per_type_and_reports_false_alert_rate(tmp_path) -> None:
    truth = tmp_path / "events.csv"
    predictions = tmp_path / "alerts.csv"
    _write(
        truth,
        "video,start_frame,end_frame,event_type,person_id\n",
        [
            "clip,10,12,rapid_movement,A",
            "clip,20,20,restricted_entry,B",
        ],
    )
    _write(
        predictions,
        "video,frame_number,track_id,event_type\n",
        [
            "clip,13,1,rapid_movement",
            "clip,13,2,rapid_movement",
            "clip,30,3,fall_like",
        ],
    )
    result = evaluate_events(truth, predictions, tolerance_frames=1, duration_seconds=60)
    assert result["rapid_movement"]["TP"] == 1
    assert result["rapid_movement"]["FP"] == 1
    assert result["restricted_entry"]["FN"] == 1
    assert result["fall_like"]["FP"] == 1
    assert result["false_alerts_per_minute"] == pytest.approx(2)


def test_line_crossing_events_score_and_report_direction_mismatches(tmp_path) -> None:
    truth = tmp_path / "line_events.csv"
    predictions = tmp_path / "line_alerts.csv"
    _write(
        truth,
        "video,start_frame,end_frame,event_type,direction\n",
        [
            "clip,10,10,line_crossing,positive_to_negative",
            "clip,20,20,line_crossing,negative_to_positive",
        ],
    )
    _write(
        predictions,
        "video,frame_number,track_id,event_type,event_detail\n",
        [
            "clip,10,track-1,line_crossing,positive_to_negative",
            "clip,20,track-2,line_crossing,positive_to_negative",
        ],
    )
    result = evaluate_events(truth, predictions, tolerance_frames=0)
    assert (result["TP"], result["FP"], result["FN"]) == (1, 1, 1)
    assert result["line_crossing"]["direction_correct"] == 1
    assert result["line_crossing"]["direction_mismatches"] == 1
    assert result["line_crossing"]["direction_accuracy"] == pytest.approx(0.5)
    assert "Implemented" in result["line_crossing_implementation"]


def test_line_crossing_direction_pairing_prefers_correct_temporal_match(
    tmp_path,
) -> None:
    truth = tmp_path / "line_events.csv"
    predictions = tmp_path / "line_alerts.csv"
    _write(
        truth,
        "video,start_frame,end_frame,event_type,direction\n",
        ["clip,10,10,line_crossing,positive_to_negative"],
    )
    _write(
        predictions,
        "video,frame_number,track_id,event_type,event_detail\n",
        [
            "clip,10,track-wrong,line_crossing,negative_to_positive",
            "clip,10,track-correct,line_crossing,positive_to_negative",
        ],
    )
    result = evaluate_events(truth, predictions, tolerance_frames=0)
    assert (result["TP"], result["FP"], result["FN"]) == (1, 1, 0)
    assert result["line_crossing"]["direction_correct"] == 1
    assert result["line_crossing"]["direction_mismatches"] == 0
    assert result["line_crossing"]["direction_accuracy"] == 1.0


def test_line_crossing_direction_is_required_and_validated(tmp_path) -> None:
    annotation = tmp_path / "events.csv"
    annotation.write_text(
        "video,start_frame,end_frame,event_type,direction\n"
        "clip,10,10,line_crossing,sideways\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="direction"):
        validate_annotations(annotation, "events")

    annotation.write_text(
        "video,start_frame,end_frame,event_type\n"
        "clip,10,10,line_crossing\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="direction"):
        validate_annotations(annotation, "events")


def test_line_crossing_direction_reads_diagnostic_event_fields(tmp_path) -> None:
    truth = tmp_path / "line_events.csv"
    predictions = tmp_path / "alerts.csv"
    diagnostics = tmp_path / "events.jsonl"
    _write(
        truth,
        "video,start_frame,end_frame,event_type,direction,start_seconds,end_seconds\n",
        ["clip,1,1,line_crossing,negative_to_positive,3.0,3.2"],
    )
    _write(
        predictions,
        "video,frame_number,track_id,event_type\n",
        ["clip,100,track-1,line_crossing"],
    )
    diagnostics.write_text(
        json.dumps(
            {
                "video_id": "clip",
                "frame_number": 100,
                "events": [
                    {
                        "event_type": "line_crossing",
                        "direction": "negative_to_positive",
                        "source_timestamp_seconds": 3.1,
                    }
                ],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    result = evaluate_events(
        truth,
        predictions,
        diagnostics_path=diagnostics,
        tolerance_seconds=0.1,
    )
    assert result["line_crossing"]["TP"] == 1
    assert result["line_crossing"]["direction_accuracy"] == 1.0


@pytest.mark.parametrize(
    ("direction", "timestamp"),
    [("sideways", 3.1), ("positive_to_negative", float("nan"))],
)
def test_event_diagnostics_reject_invalid_direction_or_source_time(
    tmp_path, direction, timestamp
) -> None:
    truth = tmp_path / "line_events.csv"
    predictions = tmp_path / "alerts.csv"
    diagnostics = tmp_path / "events.jsonl"
    _write(
        truth,
        "video,start_frame,end_frame,event_type,direction\n",
        ["clip,1,1,line_crossing,positive_to_negative"],
    )
    _write(
        predictions,
        "video,frame_number,event_type\n",
        ["clip,1,line_crossing"],
    )
    diagnostics.write_text(
        json.dumps(
            {
                "video_id": "clip",
                "frame_number": 1,
                "events": [
                    {
                        "event_type": "line_crossing",
                        "direction": direction,
                        "source_timestamp_seconds": timestamp,
                    }
                ],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="diagnostic"):
        evaluate_events(truth, predictions, diagnostics_path=diagnostics)


def test_event_matching_uses_source_timestamps_when_available(tmp_path) -> None:
    truth = tmp_path / "events.csv"
    predictions = tmp_path / "alerts.csv"
    diagnostics = tmp_path / "diagnostics.jsonl"
    _write(
        truth,
        "video,start_frame,end_frame,event_type,person_id,start_seconds,end_seconds\n",
        ["clip,1,1,fall_like,A,4.0,4.5"],
    )
    _write(
        predictions,
        "video,frame_number,track_id,event_type\n",
        ["clip,100,temporary-id,fall_like"],
    )
    diagnostics.write_text(
        json.dumps(
            {
                "video_id": "clip",
                "frame_number": 100,
                "events": [
                    {
                        "event_type": "fall_like",
                        "source_timestamp_seconds": 4.25,
                    }
                ],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    result = evaluate_events(
        truth,
        predictions,
        diagnostics_path=diagnostics,
        tolerance_seconds=0.1,
    )
    assert result["fall_like"]["TP"] == 1


def test_privacy_reports_manual_status_and_actual_mask_rectangle_coverage(tmp_path) -> None:
    annotations = tmp_path / "privacy.csv"
    diagnostics = tmp_path / "pipeline.jsonl"
    _write(
        annotations,
        "video,frame_number,face_id,x1,y1,x2,y2,mask_status,visibility\n",
        ["clip,1,F1,0,0,10,10,partial,occluded"],
    )
    diagnostics.write_text(
        json.dumps(
            {
                "video_id": "clip",
                "frame_number": 1,
                "privacy_mask_regions": [
                    {"bbox_xyxy": [0, 0, 6, 10], "method": "face"},
                    {"bbox_xyxy": [4, 0, 8, 10], "method": "fallback"},
                ],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    result = evaluate_privacy(annotations, diagnostics)
    assert result["ground_truth_faces"] == 1
    assert result["partial_face_count"] == 1
    assert result["geometric_mask_coverage_fraction"] == pytest.approx(0.8)
    assert result["unmasked_face_region_area_pixels"] == pytest.approx(20)


def test_run_metrics_fps_latency_and_resource_samples(tmp_path) -> None:
    metrics = RunMetrics("clip.mp4", source_fps=24)
    metrics.record_frame(0.1)
    metrics.record_frame(0.3)
    metrics.record_resource_sample(0, 1_000_000, warmup=True)
    metrics.record_resource_sample(50, 20_000_000)
    metrics.record_resource_sample(75, 30_000_000)
    metrics_path = tmp_path / "metrics.json"
    metrics_path.write_text(json.dumps(metrics.snapshot()), encoding="utf-8")
    result = evaluate_performance(metrics_path)
    assert result["processing_fps"] == pytest.approx(5)
    assert result["mean_latency_ms"] == pytest.approx(200)
    assert result["median_latency_ms"] == pytest.approx(200)
    assert result["p95_latency_ms"] == pytest.approx(300)
    assert result["process_resource_profile"]["sample_count"] == 2


@pytest.mark.parametrize(
    "content",
    [
        "video,frame_number,person_id,x1,y1,x2,y2\nclip,1,,4,4,2,8\n",
        "video,frame_number,x1,y1,x2,y2\nclip,1,0,0,10,10\n",
    ],
)
def test_annotation_validation_rejects_invalid_files(tmp_path, content) -> None:
    path = tmp_path / "bad.csv"
    path.write_text(content, encoding="utf-8")
    with pytest.raises(ValueError):
        validate_annotations(path, "detections")


def test_empty_annotation_file_is_valid_but_not_evaluable(tmp_path) -> None:
    path = tmp_path / "empty.csv"
    path.write_text(
        "video,frame_number,person_id,x1,y1,x2,y2\n", encoding="utf-8"
    )
    result = validate_annotations(path, "detections")
    assert result["rows"] == 0
    assert result["status"] == "valid_empty_not_evaluable"


def test_event_annotation_person_id_is_optional(tmp_path) -> None:
    path = tmp_path / "events.csv"
    _write(
        path,
        "video,start_frame,end_frame,event_type\n",
        ["clip,1,2,fall_like"],
    )
    result = validate_annotations(path, "events")
    assert result["rows"] == 1
    assert result["status"] == "valid"


def test_cli_does_not_score_header_only_ground_truth(tmp_path, capsys) -> None:
    path = tmp_path / "empty.csv"
    _write(path, "video,frame_number,person_id,x1,y1,x2,y2\n", [])
    assert evaluate_main(
        [
            "--mode",
            "detections",
            "--ground-truth",
            str(path),
            "--predictions",
            str(tmp_path / "does-not-exist.csv"),
        ]
    ) == 0
    assert "NOT EVALUATED" in capsys.readouterr().out


def test_cli_does_not_score_zero_byte_ground_truth(tmp_path, capsys) -> None:
    path = tmp_path / "empty.csv"
    path.write_text("", encoding="utf-8")
    assert evaluate_main(
        [
            "--mode",
            "detections",
            "--ground-truth",
            str(path),
            "--predictions",
            str(tmp_path / "does-not-exist.csv"),
        ]
    ) == 0
    assert "NOT EVALUATED" in capsys.readouterr().out

def test_missing_prediction_file_is_reported(tmp_path) -> None:
    truth = tmp_path / "truth.csv"
    _write(
        truth,
        "video,frame_number,person_id,x1,y1,x2,y2\n",
        ["clip,1,A,0,0,10,10"],
    )
    with pytest.raises(FileNotFoundError):
        evaluate_detections(truth, tmp_path / "missing.csv")


def test_csv_prediction_exporter_merges_real_pipeline_logs(tmp_path) -> None:
    detections = tmp_path / "detections.csv"
    tracks = tmp_path / "tracks.csv"
    alerts = tmp_path / "alerts.csv"
    output = tmp_path / "export.jsonl"
    _write(
        detections,
        "video,frame_number,person_id,x1,y1,x2,y2,confidence,class_id\n",
        ["clip,1,,0,0,10,10,0.8,0"],
    )
    _write(
        tracks,
        "video,frame_number,person_id,x1,y1,x2,y2\n",
        ["clip,1,7,0,0,10,10"],
    )
    _write(
        alerts,
        "video,frame_number,track_id,event_type\n",
        ["clip,1,7,restricted_entry"],
    )
    count = export_from_csv_logs(detections, tracks, alerts, output)
    record = json.loads(output.read_text(encoding="utf-8").splitlines()[0])
    assert count == 1
    assert record["detections"][0]["confidence"] == pytest.approx(0.8)
    assert record["tracks"][0]["track_id"] == "7"
    assert record["events"][0]["event_type"] == "restricted_entry"


def test_frame_sampling_writes_one_based_indexed_images(tmp_path) -> None:
    video = tmp_path / "sample.avi"
    writer = cv2.VideoWriter(
        str(video), cv2.VideoWriter_fourcc(*"MJPG"), 5.0, (32, 24)
    )
    assert writer.isOpened()
    for shade in (20, 40, 60, 80):
        writer.write(np.full((24, 32, 3), shade, dtype=np.uint8))
    writer.release()
    records = sample_frames(video, tmp_path / "frames", every_frames=2)
    assert [row["frame_number"] for row in records] == [1, 3]
    assert all(Path(row["image_path"]).is_file() for row in records)


def test_annotation_templates_match_validatable_schemas_and_do_not_overwrite(
    tmp_path,
) -> None:
    paths = create_templates(tmp_path / "templates")
    assert {path.stem for path in paths} == {
        "detections",
        "tracking",
        "events",
        "privacy",
    }
    for path in paths:
        annotation_type = path.stem
        result = validate_annotations(path, annotation_type)
        assert result["rows"] == 0
        assert result["status"] == "valid_empty_not_evaluable"
    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        create_templates(tmp_path / "templates")
    data_file = paths[0]
    with data_file.open("a", encoding="utf-8") as file:
        file.write("clip,1,,0,0,10,10\n")
    with pytest.raises(FileExistsError, match="contains data"):
        _open_annotation_output(data_file, HEADERS["detections"])


def test_event_annotation_interface_writes_human_entered_interval(
    tmp_path, monkeypatch
) -> None:
    import evaluation.annotate as annotator

    video = tmp_path / "clip.avi"
    video.touch()
    frames = [np.zeros((16, 16, 3), dtype=np.uint8) for _ in range(2)]

    class FakeCapture:
        def __init__(self):
            self.next_index = 0
            self.position_ms = 0.0
            self.released = False

        def isOpened(self):
            return True

        def get(self, property_id):
            if property_id == annotator.cv2.CAP_PROP_FPS:
                return 10.0
            if property_id == annotator.cv2.CAP_PROP_POS_MSEC:
                return self.position_ms
            return 0.0

        def read(self):
            if self.next_index >= len(frames):
                return False, None
            index = self.next_index
            self.next_index += 1
            self.position_ms = index * 100.0
            return True, frames[index]

        def set(self, property_id, value):
            if property_id == annotator.cv2.CAP_PROP_POS_FRAMES:
                self.next_index = int(value)
                return True
            return False

        def release(self):
            self.released = True

    capture = FakeCapture()
    key_sequence = iter((ord("["), ord("n"), ord("]"), ord("q")))
    monkeypatch.setattr(annotator.cv2, "VideoCapture", lambda _path: capture)
    monkeypatch.setattr(annotator.cv2, "resize", lambda frame, *_args, **_kwargs: frame)
    monkeypatch.setattr(annotator.cv2, "putText", lambda frame, *_args, **_kwargs: frame)
    monkeypatch.setattr(annotator.cv2, "imshow", lambda *_args: None)
    monkeypatch.setattr(
        annotator.cv2,
        "waitKey",
        lambda _wait: next(key_sequence),
    )
    monkeypatch.setattr(annotator.cv2, "destroyAllWindows", lambda: None)
    answers = iter(("line_crossing", "human-1", "positive_to_negative"))
    monkeypatch.setattr("builtins.input", lambda _prompt: next(answers))

    output_dir = tmp_path / "ground_truth"
    create_templates(output_dir)
    output = output_dir / "events.csv"
    assert annotate_video(video, output, "events") == 1
    result = validate_annotations(output, "events")
    assert result["rows"] == 1
    with output.open(newline="", encoding="utf-8") as file:
        row = next(csv.DictReader(file))
    assert row["start_frame"] == "1"
    assert row["end_frame"] == "2"
    assert row["event_type"] == "line_crossing"
    assert row["direction"] == "positive_to_negative"
    assert capture.released


def test_report_generator_writes_summary_and_svg_plot(tmp_path) -> None:
    metrics = {
        "source": "clip.mp4",
        "source_fps": 24,
        "frames_processed": 10,
        "steady_state_processing_fps": 2.5,
        "steady_state_average_frame_processing_latency_ms": 400,
        "median_frame_processing_latency_ms": 300,
        "p95_frame_processing_latency_ms": 800,
        "finalization_status": "completed",
        "run_options": {"source_resolution": [1920, 1080]},
        "steady_state_stage_profile": {
            "deepsort_update": {
                "calls": 10,
                "mean_ms": 100,
                "p95_ms": 200,
                "total_seconds": 1,
            }
        },
        "runtime_environment": {},
        "process_resource_profile": {},
    }
    metrics_path = tmp_path / "metrics.json"
    metrics_path.write_text(json.dumps(metrics), encoding="utf-8")
    report_path = generate_report([metrics_path], tmp_path / "report")
    assert report_path.is_file()
    assert (tmp_path / "report" / "run_1_stage_timings.svg").is_file()
    assert "NOT EVALUATED" in report_path.read_text(encoding="utf-8")


def test_motchallenge_conversion_scales_boxes_and_maps_odd_frames(tmp_path) -> None:
    predictions = tmp_path / "detections.csv"
    _write(
        predictions,
        "video,frame_number,person_id,x1,y1,x2,y2,confidence\n",
        [
            "MOT17-10-FRCNN,1,,1,2,11,22,0.9",
            "MOT17-10-FRCNN,3,,2,3,12,23,0.8",
        ],
    )
    output = tmp_path / "detections.txt"

    rows = _convert_prediction_csv(
        predictions,
        output,
        sequence="MOT17-10-FRCNN",
        frame_count=3,
        scale_x=2,
        scale_y=2,
        detector_sampled=True,
        tracking=False,
    )

    assert rows == 2
    assert output.read_text(encoding="utf-8").splitlines() == [
        "1,1,2.000000,4.000000,20.000000,40.000000,0.900000,1,-1,-1",
        "2,2,4.000000,6.000000,20.000000,40.000000,0.800000,1,-1,-1",
    ]


def test_motchallenge_tracking_conversion_preserves_track_identity(tmp_path) -> None:
    predictions = tmp_path / "tracks.csv"
    _write(
        predictions,
        "video,frame_number,person_id,x1,y1,x2,y2\n",
        [
            "MOT17-10-FRCNN,1,deep-sort-7,1,2,11,22",
            "MOT17-10-FRCNN,2,deep-sort-7,2,3,12,23",
        ],
    )
    output = tmp_path / "tracks.txt"

    rows = _convert_prediction_csv(
        predictions,
        output,
        sequence="MOT17-10-FRCNN",
        frame_count=2,
        scale_x=2,
        scale_y=2,
        detector_sampled=False,
        tracking=True,
    )

    lines = output.read_text(encoding="utf-8").splitlines()
    assert rows == 2
    assert [line.split(",")[:2] for line in lines] == [["1", "1"], ["2", "1"]]


def test_motchallenge_detection_conversion_rejects_even_detector_frame(tmp_path) -> None:
    predictions = tmp_path / "detections.csv"
    _write(
        predictions,
        "video,frame_number,person_id,x1,y1,x2,y2\n",
        ["MOT17-10-FRCNN,2,,1,2,11,22"],
    )

    with pytest.raises(ValueError, match="even frame"):
        _convert_prediction_csv(
            predictions,
            tmp_path / "detections.txt",
            sequence="MOT17-10-FRCNN",
            frame_count=2,
            scale_x=2,
            scale_y=2,
            detector_sampled=True,
            tracking=False,
        )


def test_motchallenge_conversion_reads_application_jsonl_export(tmp_path) -> None:
    predictions = tmp_path / "export.jsonl"
    predictions.write_text(
        json.dumps(
            {
                "video_id": "MOT17-10-FRCNN",
                "frame_number": 1,
                "detections": [
                    {
                        "bbox_xyxy": [1, 2, 11, 22],
                        "confidence": 0.9,
                        "class_id": 0,
                    }
                ],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    output = tmp_path / "detections.txt"

    rows = _convert_prediction_csv(
        predictions,
        output,
        sequence="MOT17-10-FRCNN",
        frame_count=1,
        scale_x=2,
        scale_y=2,
        detector_sampled=True,
        tracking=False,
    )

    assert rows == 1
    assert output.read_text(encoding="utf-8").strip() == (
        "1,1,2.000000,4.000000,20.000000,40.000000,0.900000,1,-1,-1"
    )


def test_motchallenge_tracking_replay_preserves_benchmark_coordinates(tmp_path) -> None:
    predictions = tmp_path / "MOT17-04-FRCNN.txt"
    predictions.write_text("1,7,10,20,30,40,1,1,-1,-1\n", encoding="utf-8")
    output = tmp_path / "replay" / predictions.name

    rows = _copy_motchallenge_tracking(
        predictions,
        output,
        frame_count=1,
    )

    assert rows == 1
    assert output.read_text(encoding="utf-8") == (
        "1,7,10,20,30,40,1,1,-1,-1\n"
    )
