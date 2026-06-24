import pytest, numpy as np
from dataclasses import asdict
from pipeline.inference import DetectionResult, HarmfulSegment

def test_detection_result_fields():
    result = DetectionResult(
        video_path="test.mp4", video_id="test_001", duration=30.0, is_harmful=True,
        anomaly_score=0.85, predicted_categories=["smoke"],
        category_scores={"smoke": 0.85, "blood": 0.1, "violent": 0.2, "abusive": 0.05, "sexy": 0.02, "money": 0.01, "policy": 0.03},
        segment_scores=np.array([0.1, 0.3, 0.85]), harmful_segments=[], keyframe_paths=[],
        attention_weights=np.array([0.1, 0.3, 0.6]), processing_time=1.5, detection_time="2026-01-01"
    )
    assert result.is_harmful == True
    assert result.anomaly_score == 0.85
    assert len(result.segment_scores) == 3

def test_harmful_segment():
    seg = HarmfulSegment(start_time=5.0, end_time=10.0, score=0.9, category="吸烟", category_en="smoke")
    assert seg.start_time == 5.0
    assert seg.category_en == "smoke"

def test_threshold_behavior():
    scores = np.array([0.1, 0.3, 0.5, 0.8, 0.9])
    threshold = 0.5
    harmful = scores >= threshold
    assert harmful.sum() == 3
