import pytest, numpy as np
from unittest.mock import patch, MagicMock

def test_class_prompts_defined():
    from pipeline.feature_extractor import CLASS_PROMPTS
    assert len(CLASS_PROMPTS) == 7
    for key in ["smoke", "blood", "violent", "abusive", "sexy", "money", "policy"]:
        assert key in CLASS_PROMPTS
        assert len(CLASS_PROMPTS[key]) >= 2

def test_similarity_computation():
    visual = np.random.randn(5, 512).astype(np.float32)
    visual = visual / np.linalg.norm(visual, axis=-1, keepdims=True)
    text = np.random.randn(3, 512).astype(np.float32)
    text = text / np.linalg.norm(text, axis=-1, keepdims=True)
    sim = visual @ text.T
    assert sim.shape == (5, 3)
    assert np.all(sim >= -1.01) and np.all(sim <= 1.01)

def test_cache_hash_consistency():
    import tempfile, os
    from pipeline.feature_extractor import CLIPFeatureExtractor
    with tempfile.NamedTemporaryFile(suffix='.npy', delete=False) as f:
        np.save(f.name, np.random.randn(10, 512))
        path = f.name
    try:
        extractor = CLIPFeatureExtractor.__new__(CLIPFeatureExtractor)
        h1 = extractor._compute_file_hash(path)
        h2 = extractor._compute_file_hash(path)
        assert h1 == h2
    finally:
        os.unlink(path)
