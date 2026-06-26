import os
import json
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import (
    precision_recall_curve,
    roc_curve,
)

from utils.tools import get_prompt_text, get_batch_mask
from utils.metrics import compute_video_level_auc, compute_video_level_ap


class Evaluator:
    def __init__(self, config, model, label_map, device=None):
        self.config = config
        self.model = model
        self.label_map = label_map
        if device is not None:
            self.device = torch.device(device)
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.visual_length = self.config["model"]["visual_length"]

    def evaluate(self, test_loader, gt):
        self.model.eval()
        prompt_text = get_prompt_text(self.label_map, self.config.get('text_prompts', {}))
        repeat_factor = 16
        ap1_list, ap2_list = [], []

        with torch.no_grad():
            for item in test_loader:
                visual = item[0]
                length = int(item[2])
                len_cur = length

                if visual.dim() == 4:
                    visual = visual.squeeze(0)
                visual = visual.to(self.device)

                num_segs = visual.shape[0]
                lengths = torch.zeros(num_segs, dtype=torch.long, device=self.device)
                remain = length
                for j in range(num_segs):
                    take = min(self.visual_length, remain)
                    lengths[j] = take
                    remain -= take

                padding_mask = get_batch_mask(lengths, self.visual_length).to(self.device)

                _, logits1, logits2, shot_slices, _ = self.model(
                    visual, padding_mask, prompt_text, lengths
                )

                prob1 = torch.sigmoid(
                    logits1.reshape(-1, 1)[:len_cur].squeeze(-1)
                )
                prob2 = (
                    1.0
                    - F.softmax(
                        logits2.reshape(-1, logits2.shape[-1])[:len_cur],
                        dim=-1,
                    )[:, 0]
                )

                ap1_list.extend(prob1.cpu().numpy().tolist())
                ap2_list.extend(prob2.cpu().numpy().tolist())

        ap1_scores = np.array(ap1_list)
        ap2_scores = np.array(ap2_list)

        final_scores_1 = np.repeat(ap1_scores, repeat_factor)
        final_scores_2 = np.repeat(ap2_scores, repeat_factor)

        gt = np.array(gt)
        gt_len = len(gt)
        if len(final_scores_1) > gt_len:
            final_scores_1 = final_scores_1[:gt_len]
        elif len(final_scores_1) < gt_len:
            final_scores_1 = np.pad(final_scores_1, (0, gt_len - len(final_scores_1)), "constant")

        if len(final_scores_2) > gt_len:
            final_scores_2 = final_scores_2[:gt_len]
        elif len(final_scores_2) < gt_len:
            final_scores_2 = np.pad(final_scores_2, (0, gt_len - len(final_scores_2)), "constant")

        try:
            auc1 = compute_video_level_auc(final_scores_1, gt)
        except ValueError:
            auc1 = 0.0
        try:
            ap1 = compute_video_level_ap(final_scores_1, gt)
        except ValueError:
            ap1 = 0.0
        try:
            auc2 = compute_video_level_auc(final_scores_2, gt)
        except ValueError:
            auc2 = 0.0
        try:
            ap2 = compute_video_level_ap(final_scores_2, gt)
        except ValueError:
            ap2 = 0.0

        fpr, tpr, _ = roc_curve(gt, final_scores_1)
        precision, recall, _ = precision_recall_curve(gt, final_scores_1)

        results = {
            "auc1": float(auc1),
            "ap1": float(ap1),
            "auc2": float(auc2),
            "ap2": float(ap2),
            "fpr": fpr.tolist(),
            "tpr": tpr.tolist(),
            "precision": precision.tolist(),
            "recall": recall.tolist(),
            "scores1": final_scores_1.tolist(),
            "scores2": final_scores_2.tolist(),
        }

        print(f"AUC1: {auc1:.4f}, AP1: {ap1:.4f}")
        print(f"AUC2: {auc2:.4f}, AP2: {ap2:.4f}")

        return results

    def evaluate_single(self, visual_features, lengths, prompt_text):
        self.model.eval()

        with torch.no_grad():
            visual_features = visual_features.to(self.device)
            if isinstance(lengths, (list, tuple)):
                lengths = torch.tensor(lengths)

            outputs = self.model(visual_features, None, prompt_text, lengths)

            if isinstance(outputs, dict):
                anomaly_score = outputs.get("anomaly_score", outputs.get("scores"))
                segment_scores = outputs.get("segment_scores", None)
                class_probs = outputs.get("class_probs", outputs.get("logits"))
                attention_weights = outputs.get("attention_weights", None)
            else:
                anomaly_score = outputs[0]
                segment_scores = outputs[1] if len(outputs) > 1 else None
                class_probs = outputs[2] if len(outputs) > 2 else None
                attention_weights = outputs[3] if len(outputs) > 3 else None

            anomaly_score_val = anomaly_score.squeeze().item()

            if segment_scores is not None:
                segment_scores_val = segment_scores.squeeze().cpu().numpy()
            else:
                segment_scores_val = np.array([anomaly_score_val])

            if class_probs is not None:
                class_probs_val = F.softmax(class_probs.squeeze(), dim=-1).cpu().numpy()
                predicted_class = int(np.argmax(class_probs_val))
            else:
                class_probs_val = np.array([1.0 - anomaly_score_val, anomaly_score_val])
                predicted_class = int(anomaly_score_val > 0.5)

            if attention_weights is not None:
                attention_weights_val = attention_weights.squeeze().cpu().numpy()
            else:
                attention_weights_val = np.ones_like(segment_scores_val)

        return (
            float(anomaly_score_val),
            np.array(segment_scores_val),
            predicted_class,
            np.array(class_probs_val),
            np.array(attention_weights_val),
        )

    def plot_roc_curve(self, fpr, tpr, auc, save_path=None):
        plt.figure()
        plt.plot(fpr, tpr, label=f"AUC = {auc:.4f}")
        plt.plot([0, 1], [0, 1], "k--")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("ROC Curve")
        plt.legend(loc="lower right")
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()

    def plot_pr_curve(self, precision, recall, ap, save_path=None):
        plt.figure()
        plt.plot(recall, precision, label=f"AP = {ap:.4f}")
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.title("Precision-Recall Curve")
        plt.legend(loc="lower left")
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()

    def export_results(self, results, output_dir):
        os.makedirs(output_dir, exist_ok=True)

        json_path = os.path.join(output_dir, "results.json")
        with open(json_path, "w") as f:
            json.dump(results, f, indent=2)

        if "fpr" in results and "tpr" in results and "auc1" in results:
            self.plot_roc_curve(
                results["fpr"], results["tpr"], results["auc1"],
                save_path=os.path.join(output_dir, "roc_curve.png"),
            )

        if "precision" in results and "recall" in results and "ap1" in results:
            self.plot_pr_curve(
                results["precision"], results["recall"], results["ap1"],
                save_path=os.path.join(output_dir, "pr_curve.png"),
            )

        report_path = os.path.join(output_dir, "report.txt")
        with open(report_path, "w") as f:
            f.write(f"AUC1: {results.get('auc1', 'N/A')}\n")
            f.write(f"AP1: {results.get('ap1', 'N/A')}\n")
            f.write(f"AUC2: {results.get('auc2', 'N/A')}\n")
            f.write(f"AP2: {results.get('ap2', 'N/A')}\n")
