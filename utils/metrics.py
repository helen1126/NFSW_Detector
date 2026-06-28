import numpy as np
from sklearn.metrics import (roc_auc_score, average_precision_score, f1_score, precision_score, recall_score, precision_recall_curve)

def compute_video_level_auc(scores, labels):
    return roc_auc_score(labels, scores)

def compute_video_level_ap(scores, labels):
    return average_precision_score(labels, scores)

def compute_video_level_f1(scores, labels, threshold=0.5):
    preds = (np.array(scores) >= threshold).astype(int)
    return f1_score(labels, preds)

def compute_class_level_metrics(scores, labels, num_classes=7, threshold=0.5):
    results = {}
    for c in range(num_classes):
        class_scores = [s[c] for s in scores]
        class_labels = [1 if l == c else 0 for l in labels]
        try:
            auc = roc_auc_score(class_labels, class_scores)
        except ValueError:
            auc = 0.0
        try:
            ap = average_precision_score(class_labels, class_scores)
        except ValueError:
            ap = 0.0
        preds = (np.array(class_scores) >= threshold).astype(int)
        f1 = f1_score(class_labels, preds, zero_division=0)
        precision = precision_score(class_labels, preds, zero_division=0)
        recall = recall_score(class_labels, preds, zero_division=0)
        results[c] = {"auc": auc, "ap": ap, "f1": f1, "precision": precision, "recall": recall}
    return results

def compute_frame_level_auc(frame_scores, frame_labels):
    return roc_auc_score(frame_labels, frame_scores)

def compute_temporal_iou(pred_segments, gt_segments):
    if not pred_segments or not gt_segments:
        return 0.0
    total_iou = 0.0
    for gt_start, gt_end in gt_segments:
        best_iou = 0.0
        for pred_start, pred_end in pred_segments:
            intersection = max(0, min(pred_end, gt_end) - max(pred_start, gt_start))
            union = max(pred_end, gt_end) - min(pred_start, gt_start)
            if union > 0:
                iou = intersection / union
                best_iou = max(best_iou, iou)
        total_iou += best_iou
    return total_iou / len(gt_segments)

def generate_summary_report(video_auc, video_ap, class_metrics, frame_auc=None, temporal_iou=None):
    report = f"Video-Level AUC: {video_auc:.4f}\n"
    report += f"Video-Level AP: {video_ap:.4f}\n"
    if frame_auc is not None:
        report += f"Frame-Level AUC: {frame_auc:.4f}\n"
    if temporal_iou is not None:
        report += f"Temporal IoU: {temporal_iou:.4f}\n"
    report += "\nPer-Class Metrics:\n"
    for c, m in class_metrics.items():
        report += f"  Class {c}: AUC={m['auc']:.4f}, AP={m['ap']:.4f}, F1={m['f1']:.4f}\n"
    return report

def serialize_metrics(metrics_dict):
    serialized = {}
    for k, v in metrics_dict.items():
        if isinstance(v, np.ndarray):
            serialized[k] = v.tolist()
        elif isinstance(v, np.floating):
            serialized[k] = float(v)
        elif isinstance(v, np.integer):
            serialized[k] = int(v)
        elif isinstance(v, dict):
            serialized[k] = serialize_metrics(v)
        else:
            serialized[k] = v
    return serialized