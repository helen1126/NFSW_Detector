import pytest, numpy as np
from utils.metrics import (
    compute_video_level_auc, compute_video_level_ap, compute_video_level_f1,
    compute_class_level_metrics, compute_temporal_iou, serialize_metrics
)

def test_video_level_auc():
    scores = np.array([0.1, 0.4, 0.6, 0.9, 0.2])
    labels = np.array([0, 0, 1, 1, 0])
    auc = compute_video_level_auc(scores, labels)
    assert 0.0 <= auc <= 1.0

def test_video_level_ap():
    scores = np.array([0.1, 0.4, 0.6, 0.9, 0.2])
    labels = np.array([0, 0, 1, 1, 0])
    ap = compute_video_level_ap(scores, labels)
    assert 0.0 <= ap <= 1.0

def test_video_level_f1():
    scores = np.array([0.1, 0.4, 0.6, 0.9, 0.2])
    labels = np.array([0, 0, 1, 1, 0])
    f1 = compute_video_level_f1(scores, labels, threshold=0.5)
    assert 0.0 <= f1 <= 1.0

def test_class_level_metrics():
    scores = [[0.1]*7, [0.9]*7, [0.5]*7, [0.3]*7, [0.8]*7, [0.2]*7, [0.6]*7]
    labels = [0, 1, 2, 3, 4, 5, 6]
    results = compute_class_level_metrics(scores, labels, num_classes=7)
    assert len(results) == 7
    for c, m in results.items():
        assert "auc" in m
        assert "ap" in m
        assert "f1" in m

def test_temporal_iou():
    pred = [(5.0, 15.0)]
    gt = [(10.0, 20.0)]
    iou = compute_temporal_iou(pred, gt)
    assert 0.0 <= iou <= 1.0

def test_serialize_metrics():
    data = {"auc": np.float64(0.85), "scores": np.array([0.1, 0.2])}
    serialized = serialize_metrics(data)
    assert isinstance(serialized["auc"], float)
    assert isinstance(serialized["scores"], list)
