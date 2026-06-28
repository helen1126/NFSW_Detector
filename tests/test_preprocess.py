import pytest, numpy as np
from pipeline.preprocess import VideoPreprocessor

def test_uniform_sample(sample_config):
    preprocessor = VideoPreprocessor(sample_config)
    frames = np.random.randint(0, 255, (100, 224, 224, 3), dtype=np.uint8)
    indices = preprocessor.uniform_sample(frames, 10)
    assert len(indices) == 10
    assert indices.min() >= 0 and indices.max() < 100
    sampled = frames[indices]
    assert sampled.shape == (10, 224, 224, 3)

def test_scene_keyframes(sample_config):
    preprocessor = VideoPreprocessor(sample_config)
    frames = np.random.randint(0, 255, (50, 224, 224, 3), dtype=np.uint8)
    frames[25:] = frames[25:] + 100
    keyframe_indices = preprocessor.scene_keyframes(frames, threshold=10.0)
    assert isinstance(keyframe_indices, (list, np.ndarray))

def test_validate_format_unsupported(sample_config):
    preprocessor = VideoPreprocessor(sample_config)
    with pytest.raises(ValueError, match="Unsupported"):
        preprocessor.validate_format("test.txt")

def test_validate_format_missing_file(sample_config):
    preprocessor = VideoPreprocessor(sample_config)
    with pytest.raises(ValueError, match="not found"):
        preprocessor.validate_format("nonexistent.mp4")

def test_video_meta_info_computation(sample_config):
    preprocessor = VideoPreprocessor(sample_config)
    assert preprocessor.supported_formats == ["mp4", "avi", "mov", "flv", "mkv", "wmv", "webm"]
