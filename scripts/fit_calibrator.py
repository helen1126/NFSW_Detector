"""离线校准脚本：在 train.csv 上跑推理收集 (raw_score, label)，拟合 IsotonicRegression。

用法：
    python scripts/fit_calibrator.py --config configs/default.yaml \
        --checkpoint checkpoints/best_model.pth --output checkpoints/calibrator.pkl

校准数据来源：train.csv（含 normal + abnormal 两类样本），不污染 test.csv。
"""
import argparse
import os
import sys
import numpy as np
import pandas as pd
import yaml

# 添加项目根到 path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pipeline.inference import NSFWDetector
from pipeline.calibration import ScoreCalibrator


def main():
    parser = argparse.ArgumentParser(description="拟合分数校准器")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--checkpoint", required=True, help="模型权重路径")
    parser.add_argument("--output", default="checkpoints/calibrator.pkl", help="校准器输出路径")
    parser.add_argument("--max-samples", type=int, default=None,
                        help="最多使用的样本数（按类别均衡采样），不传则用全部")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    train_csv = config["data"]["train_csv"]
    df = pd.read_csv(train_csv)
    print(f"加载 train.csv: {len(df)} 条样本")
    print(f"  normal: {(df['label'] == 'normal').sum()}")
    print(f"  abnormal: {(df['label'] != 'normal').sum()}")

    # 类别均衡采样
    if args.max_samples is not None:
        normal_df = df[df["label"] == "normal"].sample(
            min(args.max_samples // 2, (df["label"] == "normal").sum()),
            random_state=42
        )
        abnormal_df = df[df["label"] != "normal"].sample(
            min(args.max_samples // 2, (df["label"] != "normal").sum()),
            random_state=42
        )
        df = pd.concat([normal_df, abnormal_df]).reset_index(drop=True)
        print(f"均衡采样后: {len(df)} 条")

    # 临时关闭校准避免循环依赖
    config.setdefault("calibration", {})["enabled"] = False
    detector = NSFWDetector(config, checkpoint_path=args.checkpoint)

    raw_scores = []
    labels = []
    skipped = 0
    for i, row in df.iterrows():
        feat_path = row["path"]
        if not os.path.exists(feat_path):
            skipped += 1
            continue
        try:
            # 直接用 .npy 特征推理（绕过视频解码）
            result = _detect_from_npy(detector, feat_path)
            raw_scores.append(result.anomaly_score)
            labels.append(0 if row["label"] == "normal" else 1)
        except Exception as e:
            print(f"  [WARN] 样本 {feat_path} 推理失败: {e}")
            skipped += 1

    print(f"\n收集到 {len(raw_scores)} 条 (raw_score, label)，跳过 {skipped} 条")
    if len(raw_scores) < 10:
        print("[ERROR] 样本数过少，无法拟合校准器")
        sys.exit(1)

    raw_scores = np.array(raw_scores)
    labels = np.array(labels)
    print(f"normal 平均 raw_score: {raw_scores[labels == 0].mean():.4f}")
    print(f"abnormal 平均 raw_score: {raw_scores[labels == 1].mean():.4f}")

    calibrator = ScoreCalibrator()
    calibrator.fit(raw_scores, labels)
    calibrator.save(args.output)
    print(f"\n校准器已保存: {args.output}")

    # 输出校准前后对比
    print("\n校准前后对比（采样）:")
    print(f"  {'raw':>8} -> {'calibrated':>10}")
    for raw in [0.1, 0.3, 0.5, 0.7, 0.9]:
        cal = calibrator.transform(raw)
        print(f"  {raw:>8.2f} -> {cal:>10.4f}")


def _detect_from_npy(detector, npy_path):
    """直接从 .npy 特征文件跑推理，绕过视频解码。

    SVA 训练数据是预提取特征，此函数复用 NSFWDetector 的模型推理逻辑。
    """
    import torch
    from utils.tools import process_feat, get_prompt_text

    features = np.load(npy_path)
    visual_length = detector.model.visual_length
    features, valid_length = process_feat(features, visual_length)

    with torch.inference_mode():
        features_tensor = torch.tensor(features, dtype=torch.float32, device=detector.device)
        features_tensor = features_tensor.unsqueeze(0)
        S = features_tensor.shape[1]
        lengths = torch.tensor([valid_length], dtype=torch.long)
        padding_mask = torch.arange(S, device=detector.device).unsqueeze(0) >= lengths.unsqueeze(1).to(detector.device)
        text_list = get_prompt_text(
            detector.config.get('label_map', {}),
            detector.config.get('text_prompts', {})
        )
        _, logits1, logits2, shot_slices, attn_weights = detector.model(
            features_tensor, padding_mask, text_list, lengths
        )
        segment_scores = torch.sigmoid(logits1[:, 0]).cpu().numpy()

    valid_scores = segment_scores.reshape(-1)[:valid_length]
    anomaly_score = float(np.max(valid_scores)) if valid_length > 0 else 0.0

    # 构造最小化 DetectionResult
    from pipeline.inference import DetectionResult
    return DetectionResult(
        video_path=npy_path, video_id=os.path.splitext(os.path.basename(npy_path))[0],
        duration=0.0, is_harmful=anomaly_score >= detector.threshold,
        anomaly_score=anomaly_score, predicted_categories=[],
        category_scores={}, segment_scores=segment_scores,
        harmful_segments=[], keyframe_paths=[],
        attention_weights=np.zeros(S), processing_time=0.0,
        detection_time=""
    )


if __name__ == "__main__":
    main()
