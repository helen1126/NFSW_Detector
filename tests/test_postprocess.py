"""推理增强单元测试：校准 / OOD / 质量加权 / 零样本 / 采样帧数 / 图片检测。"""
import os
import tempfile
import numpy as np
import pytest

from pipeline.calibration import ScoreCalibrator
from pipeline.inference import NSFWDetector, DetectionResult
from pipeline.preprocess import VideoPreprocessor


# ==================== ScoreCalibrator ====================

def test_calibrator_fit_transform():
    """拟合后 transform 应单调递增。"""
    cal = ScoreCalibrator()
    raw = np.array([0.1, 0.3, 0.5, 0.7, 0.9])
    labels = np.array([0, 0, 1, 1, 1])
    cal.fit(raw, labels)
    assert cal.fitted
    out = [cal.transform(r) for r in raw]
    # 单调递增
    for i in range(len(out) - 1):
        assert out[i] <= out[i + 1], f"非单调: {out}"
    # 范围 [0,1]
    assert all(0.0 <= v <= 1.0 for v in out)


def test_calibrator_unfitted_passthrough():
    """未拟合时 transform 返回原值。"""
    cal = ScoreCalibrator()
    assert not cal.fitted
    assert cal.transform(0.42) == 0.42
    arr = np.array([0.1, 0.5, 0.9])
    # transform_batch 返回 float32，允许微小精度差异
    np.testing.assert_allclose(cal.transform_batch(arr), arr, rtol=1e-6)


def test_calibrator_save_load(tmp_path):
    """保存后加载应保持一致性。"""
    cal = ScoreCalibrator()
    raw = np.array([0.1, 0.2, 0.8, 0.9])
    labels = np.array([0, 0, 1, 1])
    cal.fit(raw, labels)
    p = str(tmp_path / "cal.pkl")
    cal.save(p)
    assert os.path.exists(p)
    cal2 = ScoreCalibrator.load(p)
    assert cal2.fitted
    assert abs(cal.transform(0.5) - cal2.transform(0.5)) < 1e-6


def test_calibrator_load_missing_file():
    """文件不存在时返回未拟合实例。"""
    cal = ScoreCalibrator.load("nonexistent_calibrator.pkl")
    assert not cal.fitted


# ==================== _compute_ood_score ====================

def _make_detector_stub():
    """创建未初始化的 NSFWDetector（跳过模型加载）。"""
    return NSFWDetector.__new__(NSFWDetector)


def test_compute_ood_score_normal():
    """normal 概率高时 ood_score 应低。"""
    det = _make_detector_stub()
    # 8 类，normal 概率 0.95，其余均分
    probs = np.zeros((10, 8))
    probs[:, 0] = 0.95
    probs[:, 1:] = 0.05 / 7
    score = det._compute_ood_score(probs, 10)
    assert 0.0 <= score <= 1.0
    assert score < 0.3, f"normal 视频应低 OOD 分数, got {score}"


def test_compute_ood_score_ood():
    """均匀分布时 ood_score 应高。"""
    det = _make_detector_stub()
    # 8 类均匀分布
    probs = np.ones((10, 8)) / 8.0
    score = det._compute_ood_score(probs, 10)
    assert score > 0.5, f"均匀分布应高 OOD 分数, got {score}"


def test_compute_ood_score_empty():
    """空输入返回中性 0.5。"""
    det = _make_detector_stub()
    assert det._compute_ood_score(np.zeros((0, 8)), 0) == 0.5


# ==================== _compute_frame_quality ====================

def test_frame_quality_clear_vs_blurry():
    """清晰帧 quality 应高于模糊帧。"""
    det = _make_detector_stub()
    # 清晰帧：随机噪声（高频）
    clear = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
    # 模糊帧：纯色（无高频）
    blurry = np.full((64, 64, 3), 128, dtype=np.uint8)
    frames = np.array([clear, blurry])
    qualities = det._compute_frame_quality(frames)
    assert qualities.shape == (2,)
    assert qualities[0] > qualities[1], f"清晰帧应更高: {qualities}"


def test_frame_quality_range():
    """质量分数应在 [0,1]。"""
    det = _make_detector_stub()
    frames = np.random.randint(0, 255, (5, 32, 32, 3), dtype=np.uint8)
    qualities = det._compute_frame_quality(frames)
    assert all(0.0 <= q <= 1.0 for q in qualities)


# ==================== _init_extra_categories ====================

class _MockFeatureExtractor:
    """模拟 CLIPFeatureExtractor，避免加载真实 CLIP 模型。"""
    def __init__(self, dim=512):
        self.dim = dim

    def extract_text_features(self, prompts):
        # 返回归一化的随机文本特征 [num_prompts, dim]
        feats = np.random.randn(len(prompts), self.dim).astype(np.float32)
        feats = feats / np.linalg.norm(feats, axis=-1, keepdims=True)
        return feats


def test_init_extra_categories():
    """初始化后 self.extra_categories 应包含配置的类别。"""
    det = _make_detector_stub()
    det.feature_extractor = _MockFeatureExtractor()
    cfg = {
        "gambling": {"prompts": ["people gambling", "casino games"], "zh": "赌博"},
        "drug": {"prompts": ["drug use", "substance abuse"], "zh": "毒品"},
    }
    det._init_extra_categories(cfg)
    assert "gambling" in det.extra_categories
    assert "drug" in det.extra_categories
    assert det.extra_categories["gambling"]["zh"] == "赌博"
    assert det.extra_categories["gambling"]["text_feat"].shape == (512,)


def test_init_extra_categories_empty_prompts_skipped():
    """空 prompts 列表应被跳过。"""
    det = _make_detector_stub()
    det.feature_extractor = _MockFeatureExtractor()
    cfg = {"empty_cat": {"prompts": [], "zh": "空"}}
    det._init_extra_categories(cfg)
    assert "empty_cat" not in det.extra_categories


# ==================== preprocess num_segments 覆盖 ====================

def test_preprocess_num_segments_override(sample_config):
    """num_segments 参数应覆盖 config 默认值。"""
    preprocessor = VideoPreprocessor(sample_config)
    # uniform_sample 直接测试
    frames = np.random.randint(0, 255, (100, 224, 224, 3), dtype=np.uint8)
    # num_segments=20
    indices = preprocessor.uniform_sample(frames, 20)
    assert len(indices) == 20
    # num_segments=5
    indices = preprocessor.uniform_sample(frames, 5)
    assert len(indices) == 5


# ==================== DetectionResult 字段 ====================

def test_detection_result_new_fields_default():
    """新字段应有默认值保证向后兼容。"""
    result = DetectionResult(
        video_path="test.mp4", video_id="test", duration=10.0,
        is_harmful=False, anomaly_score=0.2, predicted_categories=[],
        category_scores={}, segment_scores=np.array([0.1, 0.2]),
        harmful_segments=[], keyframe_paths=[],
        attention_weights=np.array([0.5]), processing_time=0.1,
        detection_time="2026-01-01 00:00:00",
    )
    assert result.calibrated_score == 0.0
    assert result.ood_score == 0.0
    assert result.is_ood is False
    assert result.extra_category_info == {}


def test_detection_result_new_fields_set():
    """新字段应可设置。"""
    result = DetectionResult(
        video_path="test.mp4", video_id="test", duration=10.0,
        is_harmful=True, anomaly_score=0.8, predicted_categories=["Smoke"],
        category_scores={"Smoke": 0.8}, segment_scores=np.array([0.8]),
        harmful_segments=[], keyframe_paths=[],
        attention_weights=np.array([0.5]), processing_time=0.1,
        detection_time="2026-01-01 00:00:00",
        calibrated_score=0.82, ood_score=0.15, is_ood=False,
        extra_category_info={"Gambling": {"zh": "赌博", "score": 0.3, "is_extra": True}},
    )
    assert result.calibrated_score == 0.82
    assert result.ood_score == 0.15
    assert result.is_ood is False
    assert "Gambling" in result.extra_category_info
