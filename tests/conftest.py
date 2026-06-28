import pytest, numpy as np, torch, os, tempfile, yaml

@pytest.fixture
def sample_config():
    return {
        "model": {"name": "SVLA", "clip_variant": "ViT-B/16", "embed_dim": 512, "visual_width": 512,
                  "visual_length": 256, "visual_head": 1, "visual_layers": 2, "attn_window": 8,
                  "prompt_prefix": 10, "prompt_postfix": 10, "hidden_dim": 256, "num_classes": 7,
                  "num_classes_with_normal": 8, "dropout": 0.3, "shot_condition": True,
                  "shot_sim_thresh": 0.90, "shot_min_len": 12, "shot_layers": 1, "shot_gamma": 0.05,
                  "pi_floor": 0.05, "cfa_tau": 0.8, "cfa_beta": 0.8, "cfa_prefix_len": 32,
                  "cfa_bottleneck": 256, "cfa_prefix_rank": 16, "cfa_dropout": 0.1},
        "data": {"dataset": "SVA", "feature_dir": "data/features", "train_csv": "data/splits/train.csv",
                 "test_csv": "data/splits/test.csv", "num_segments": 10, "frame_sample_rate": 1,
                 "supported_formats": ["mp4", "avi", "mov", "flv", "mkv"]},
        "training": {"batch_size": 32, "lr": 0.0001, "weight_decay": 0.0005, "epochs": 50,
                     "scheduler": "cosine", "warmup_epochs": 5, "loss": "bce", "class_loss_alpha": 1.0,
                     "checkpoint_dir": "checkpoints", "log_interval": 10, "early_stop_patience": 10,
                     "pi_lr_mult": 5.0, "txtreg_weight": 0.1},
        "cuda": {"device": "cuda:0", "num_workers": 4, "pin_memory": True, "cudnn_benchmark": True,
                 "deterministic": False, "amp": True, "tf32": True},
        "inference": {"anomaly_threshold": 0.5, "alert_levels": {"high": 0.8, "medium": 0.5, "low": 0.3},
                      "max_duration": 300, "device_priority": "gpu", "fallback_cpu": True},
        "demo": {"port": 7860, "share": False, "max_file_size": 500, "show_progress": True},
        "logging": {"level": "INFO", "log_file": "logs/nfsw_detector.log", "tensorboard": True, "tensorboard_dir": "runs"},
        "labels": {0: {"en": "Smoke", "zh": "吸烟"}, 1: {"en": "Blood", "zh": "血腥"},
                   2: {"en": "Violent", "zh": "暴力"}, 3: {"en": "Abusive", "zh": "辱骂"},
                   4: {"en": "Sexy", "zh": "色情"}, 5: {"en": "Money", "zh": "金钱诈骗"},
                   6: {"en": "Policy", "zh": "政治敏感"}},
        "label_map": {"normal": "normal", "smoke": "smoke", "blood": "blood", "violent": "violent",
                      "abusive": "abusive", "sexy": "sexy", "money": "money", "policy": "policy"}
    }

@pytest.fixture
def sample_features():
    return np.random.randn(256, 512).astype(np.float32)

@pytest.fixture
def sample_detection_result():
    from pipeline.inference import DetectionResult, HarmfulSegment
    return DetectionResult(
        video_path="test.mp4", video_id="test_001", duration=30.0, is_harmful=True,
        anomaly_score=0.85, predicted_categories=["smoke", "violent"],
        category_scores={"smoke": 0.85, "blood": 0.1, "violent": 0.7, "abusive": 0.05, "sexy": 0.02, "money": 0.01, "policy": 0.03},
        segment_scores=np.array([0.1, 0.3, 0.85, 0.9, 0.7, 0.2, 0.1, 0.05, 0.3, 0.6]),
        harmful_segments=[HarmfulSegment(start_time=6.0, end_time=18.0, score=0.9, category="吸烟", category_en="smoke")],
        keyframe_paths=[], attention_weights=np.random.rand(10),
        processing_time=1.5, detection_time="2026-01-01 12:00:00"
    )
