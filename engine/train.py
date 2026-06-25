import os
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import MultiStepLR
from sklearn.metrics import roc_auc_score, average_precision_score
from torch.utils.tensorboard import SummaryWriter

from utils.tools import get_prompt_text, get_batch_label, get_batch_mask


def _pi_topk_segment(scores_seg: torch.Tensor,
                     pi: torch.Tensor = None,
                     base_div: int = 16,
                     k_min: int = 1) -> torch.Tensor:
    L = scores_seg.numel()
    if L == 0:
        return scores_seg.new_tensor(0.0)

    k_base = max(1, int(L / base_div) + 1)

    if pi is not None:
        pi_val = float(pi.detach().clamp(1e-3, 0.9).item())
        pi0 = 0.22
        gamma = 2.0
        scale = 1.0 + gamma * (pi_val - pi0)
        scale = max(0.5, min(2.0, scale))
        k = int(round(k_base * scale))
    else:
        k = k_base

    k = max(k_min, k)
    k = min(L, k)
    k = max(1, k)

    topk_vals, _ = torch.topk(scores_seg, k=k, largest=True)
    return topk_vals.mean()


def CLAS2_dasmil_weighted(
    logits: torch.Tensor,
    labels: torch.Tensor,
    lengths: torch.Tensor,
    shot_slices,
    shot_pi_list,
    device,
    base_div: int = 16,
    k_min: int = 1,
    focal: bool = False,
    focal_alpha: float = 0.25,
    focal_gamma: float = 2.0,
):
    B, T, _ = logits.shape
    labels_bin = 1 - labels[:, 0].reshape(labels.shape[0]).to(device)
    probs = torch.sigmoid(logits).reshape(B, T)

    instance_logits = []
    for i in range(B):
        Li = int(lengths[i].item())
        if Li <= 0:
            instance_logits.append(probs.new_tensor(0.0).unsqueeze(0))
            continue

        shots = shot_slices[i] if (shot_slices is not None and i < len(shot_slices)) else [(0, Li)]
        pis = shot_pi_list[i] if (shot_pi_list is not None and i < len(shot_pi_list)) else None

        shot_vals = []
        used_pis = []

        for si, (s, e) in enumerate(shots):
            s = max(0, min(s, Li))
            e = max(0, min(e, Li))
            if e <= s:
                continue

            seg = probs[i, s:e]
            pi = pis[si] if (pis is not None and si < len(pis)) else None

            pooled = _pi_topk_segment(seg, pi, base_div=base_div, k_min=k_min)
            shot_vals.append(pooled)
            if pi is not None:
                used_pis.append(pi)

        if len(shot_vals) > 0:
            shot_scores = torch.stack(shot_vals, dim=0)
            if len(used_pis) == len(shot_vals) and len(used_pis) > 0:
                pis_used = torch.stack(used_pis, dim=0)
                w = pis_used / (pis_used.sum() + 1e-6)
                val = (shot_scores * w).sum()
            else:
                val = shot_scores.mean()
        else:
            val = probs[i, :Li].mean()

        instance_logits.append(val.unsqueeze(0))

    instance_logits = torch.cat(instance_logits, dim=0)
    if focal:
        bce = F.binary_cross_entropy(instance_logits, labels_bin, reduction='none')
        pt = torch.exp(-bce)
        focal_weight = focal_alpha * (1 - pt) ** focal_gamma
        clsloss = (focal_weight * bce).mean()
    else:
        clsloss = F.binary_cross_entropy(instance_logits, labels_bin)
    return clsloss


def CLASM_dasmil_weighted(
    logits: torch.Tensor,
    labels: torch.Tensor,
    lengths: torch.Tensor,
    shot_slices,
    shot_pi_list,
    device,
    base_div: int = 16,
    k_min: int = 1
):
    B, T, C = logits.shape
    labels = labels / torch.sum(labels, dim=1, keepdim=True)
    labels = labels.to(device)

    instance_logits = []
    for i in range(B):
        Li = int(lengths[i].item())
        if Li <= 0:
            instance_logits.append(logits.new_zeros(1, C))
            continue

        shots = shot_slices[i] if (shot_slices is not None and i < len(shot_slices)) else [(0, Li)]
        pis = shot_pi_list[i] if (shot_pi_list is not None and i < len(shot_pi_list)) else None

        shot_vecs = []
        used_pis = []

        for si, (s, e) in enumerate(shots):
            s = max(0, min(s, Li))
            e = max(0, min(e, Li))
            if e <= s:
                continue

            seg_logits = logits[i, s:e, :]
            pi = pis[si] if (pis is not None and si < len(pis)) else None

            pooled_per_class = []
            for c in range(C):
                seg_c = seg_logits[:, c]
                pooled_c = _pi_topk_segment(seg_c, pi, base_div=base_div, k_min=k_min)
                pooled_per_class.append(pooled_c)
            shot_vec = torch.stack(pooled_per_class, dim=0)
            shot_vecs.append(shot_vec)
            if pi is not None:
                used_pis.append(pi)

        if len(shot_vecs) > 0:
            shot_mat = torch.stack(shot_vecs, dim=0)
            if len(used_pis) == len(shot_vecs) and len(used_pis) > 0:
                pis_used = torch.stack(used_pis, dim=0).view(-1, 1)
                w = pis_used / (pis_used.sum() + 1e-6)
                vid_vec = (shot_mat * w).sum(dim=0)
            else:
                vid_vec = shot_mat.mean(dim=0)
        else:
            vid_vec = logits[i, :Li, :].mean(dim=0)

        instance_logits.append(vid_vec.unsqueeze(0))

    instance_logits = torch.cat(instance_logits, dim=0)
    milloss = -torch.mean(
        torch.sum(labels * F.log_softmax(instance_logits, dim=1), dim=1),
        dim=0,
    )
    return milloss


def text_feature_regularizer(text_features: torch.Tensor,
                             weight: float = 1e-1,
                             eps: float = 1e-12):
    if text_features is None or text_features.ndim != 2 or text_features.size(0) < 2:
        device = text_features.device if isinstance(text_features, torch.Tensor) else "cpu"
        return torch.zeros(1, device=device)

    tf = text_features
    tf = tf / (tf.norm(dim=-1, keepdim=True) + eps)
    normal = tf[0]
    others = tf[1:]
    cos = torch.matmul(others, normal)
    loss = torch.abs(cos).mean() * weight
    return loss


class Trainer:
    def __init__(self, config, model, label_map, device=None):
        self.config = config
        self.label_map = label_map
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        self.model = model.to(self.device)

        if torch.cuda.is_available():
            cuda_cfg = self.config.get("cuda", {})
            torch.backends.cudnn.benchmark = cuda_cfg.get("cudnn_benchmark", True)
            torch.backends.cudnn.deterministic = cuda_cfg.get("deterministic", False)
            if cuda_cfg.get("tf32", True):
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True
            gpu_name = torch.cuda.get_device_name(0)
            gpu_mem = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
            cuda_ver = torch.version.cuda
            print(f"GPU: {gpu_name}, Memory: {gpu_mem:.1f} GB, CUDA: {cuda_ver}")

        self.amp = self.config.get("cuda", {}).get("amp", True)
        self.scaler = torch.amp.GradScaler('cuda', enabled=self.amp)
        self.optimizer = self._setup_optimizer()
        self.scheduler = self._setup_scheduler(self.optimizer)
        self.writer = SummaryWriter(log_dir=self.config.get("logging", {}).get("tensorboard_dir", "runs"))

        train_cfg = self.config["training"]
        self.visual_length = self.config["model"]["visual_length"]
        self.txtreg_weight = float(train_cfg.get("txtreg_weight", 1e-1))
        self.pi_topk_base_div = int(train_cfg.get("pi_topk_base_div", 16))
        self.pi_topk_k_min = int(train_cfg.get("pi_topk_k_min", 1))
        self.log_interval = int(train_cfg.get("log_interval", 10))
        self.class_loss_alpha = float(train_cfg.get("class_loss_alpha", 1.0))
        self.focal = bool(train_cfg.get("focal", False))
        self.focal_alpha = float(train_cfg.get("focal_alpha", 0.25))
        self.focal_gamma = float(train_cfg.get("focal_gamma", 2.0))

    def _setup_optimizer(self):
        train_cfg = self.config["training"]
        lr = train_cfg["lr"]
        weight_decay = train_cfg["weight_decay"]
        pi_lr_mult = train_cfg.get("pi_lr_mult", 5.0)

        pi_params = [p for n, p in self.model.named_parameters()
                     if n.startswith("shot_density_head") and p.requires_grad]
        other_params = [p for n, p in self.model.named_parameters()
                        if not n.startswith("shot_density_head") and p.requires_grad]

        param_groups = [
            {"params": other_params, "lr": lr, "weight_decay": weight_decay},
            {"params": pi_params, "lr": lr * pi_lr_mult, "weight_decay": 0.0},
        ]
        optimizer = torch.optim.AdamW(param_groups)
        return optimizer

    def _setup_scheduler(self, optimizer):
        train_cfg = self.config["training"]
        milestones = train_cfg.get("scheduler_milestones", [4, 8])
        gamma = train_cfg.get("scheduler_rate", 0.1)
        scheduler = MultiStepLR(optimizer, milestones=milestones, gamma=gamma)
        return scheduler

    def train_epoch(self, normal_loader, anomaly_loader, epoch):
        self.model.train()
        total_loss = 0.0
        total_loss1 = 0.0
        total_loss2 = 0.0
        total_loss3 = 0.0
        num_steps = min(len(normal_loader), len(anomaly_loader))

        normal_iter = iter(normal_loader)
        anomaly_iter = iter(anomaly_loader)

        for step in range(num_steps):
            try:
                normal_batch = next(normal_iter)
            except StopIteration:
                normal_iter = iter(normal_loader)
                normal_batch = next(normal_iter)
            try:
                anomaly_batch = next(anomaly_iter)
            except StopIteration:
                anomaly_iter = iter(anomaly_loader)
                anomaly_batch = next(anomaly_iter)

            self.optimizer.zero_grad()

            if self.amp:
                with torch.amp.autocast('cuda'):
                    loss, loss1, loss2, loss3 = self._compute_loss(normal_batch, anomaly_batch)
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss, loss1, loss2, loss3 = self._compute_loss(normal_batch, anomaly_batch)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()

            total_loss += loss.item()
            total_loss1 += float(loss1)
            total_loss2 += float(loss2)
            total_loss3 += float(loss3)

            if (step + 1) % self.log_interval == 0:
                avg_loss = total_loss / (step + 1)
                if torch.cuda.is_available():
                    mem_used = torch.cuda.max_memory_allocated() / (1024 ** 3)
                    print(f"Epoch [{epoch}] Step [{step+1}/{num_steps}] Loss: {avg_loss:.4f} "
                          f"(bin: {total_loss1/(step+1):.4f} cls: {total_loss2/(step+1):.4f} "
                          f"txt: {total_loss3/(step+1):.4f}) GPU Mem: {mem_used:.2f} GB")
                else:
                    print(f"Epoch [{epoch}] Step [{step+1}/{num_steps}] Loss: {avg_loss:.4f}")

        return total_loss / num_steps

    def _compute_loss(self, normal_batch, anomaly_batch):
        normal_feats, normal_labels, normal_lengths = normal_batch
        anomaly_feats, anomaly_labels, anomaly_lengths = anomaly_batch

        feats = torch.cat([normal_feats, anomaly_feats], dim=0).to(self.device)
        text_labels_raw = list(normal_labels) + list(anomaly_labels)
        lengths = torch.cat([normal_lengths, anomaly_lengths], dim=0).to(self.device)

        text_list = get_prompt_text(self.label_map)
        text_labels = get_batch_label(text_labels_raw, text_list, self.label_map).to(self.device)

        text_features, logits1, logits2, shot_slices, _ = self.model(
            feats, None, text_list, lengths
        )
        shot_pi_list = getattr(self.model, "_last_shot_pi_list", None)

        loss1 = CLAS2_dasmil_weighted(
            logits1, text_labels, lengths, shot_slices, shot_pi_list,
            device=self.device, base_div=self.pi_topk_base_div, k_min=self.pi_topk_k_min,
            focal=self.focal, focal_alpha=self.focal_alpha, focal_gamma=self.focal_gamma,
        )
        loss2 = CLASM_dasmil_weighted(
            logits2, text_labels, lengths, shot_slices, shot_pi_list,
            device=self.device, base_div=self.pi_topk_base_div, k_min=self.pi_topk_k_min,
        )
        loss3 = text_feature_regularizer(text_features, weight=self.txtreg_weight)

        loss = loss1 + self.class_loss_alpha * loss2 + loss3
        return loss, loss1, loss2, loss3

    def validate(self, test_loader, gt):
        self.model.eval()
        prompt_text = get_prompt_text(self.label_map)
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
            auc1 = roc_auc_score(gt, final_scores_1)
        except ValueError:
            auc1 = 0.0
        try:
            ap1 = average_precision_score(gt, final_scores_1)
        except ValueError:
            ap1 = 0.0
        try:
            auc2 = roc_auc_score(gt, final_scores_2)
        except ValueError:
            auc2 = 0.0
        try:
            ap2 = average_precision_score(gt, final_scores_2)
        except ValueError:
            ap2 = 0.0

        return auc1, ap1, auc2, ap2

    def train(self, normal_loader, anomaly_loader, test_loader, gt):
        train_cfg = self.config["training"]
        epochs = train_cfg["epochs"]
        early_stop_patience = train_cfg.get("early_stop_patience", 10)
        best_auc1 = -1.0
        best_ap1 = -1.0
        best_auc2 = -1.0
        best_ap2 = -1.0
        patience_counter = 0
        best_metrics = {}

        for epoch in range(epochs):
            avg_loss = self.train_epoch(normal_loader, anomaly_loader, epoch)
            auc1, ap1, auc2, ap2 = self.validate(test_loader, gt)

            self.writer.add_scalar("Loss/train", avg_loss, epoch)
            self.writer.add_scalar("AUC1/val", auc1, epoch)
            self.writer.add_scalar("AP1/val", ap1, epoch)
            self.writer.add_scalar("AUC2/val", auc2, epoch)
            self.writer.add_scalar("AP2/val", ap2, epoch)

            print(f"Epoch [{epoch}/{epochs}] Loss: {avg_loss:.4f} "
                  f"AUC1: {auc1:.4f} AP1: {ap1:.4f} | AUC2: {auc2:.4f} AP2: {ap2:.4f}")

            improved = False
            if auc1 > best_auc1:
                best_auc1 = auc1
                improved = True
            if ap1 > best_ap1:
                best_ap1 = ap1
                improved = True
            if auc2 > best_auc2:
                best_auc2 = auc2
                improved = True
            if ap2 > best_ap2:
                best_ap2 = ap2
                improved = True

            if improved:
                patience_counter = 0
                best_metrics = {
                    "auc1": best_auc1, "ap1": best_ap1,
                    "auc2": best_auc2, "ap2": best_ap2, "epoch": epoch,
                }
                self._save_checkpoint(epoch, is_best=True)
            else:
                patience_counter += 1

            self._save_checkpoint(epoch, is_best=False)

            if patience_counter >= early_stop_patience:
                print(f"Early stopping at epoch {epoch}, no improvement for {early_stop_patience} epochs")
                break

            self.scheduler.step()

        self.writer.close()
        return best_metrics

    def _save_checkpoint(self, epoch, is_best=False):
        ckpt_dir = self.config["training"].get("checkpoint_dir", "checkpoints")
        os.makedirs(ckpt_dir, exist_ok=True)
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "config": self.config,
        }
        path = os.path.join(ckpt_dir, f"checkpoint_epoch_{epoch}.pth")
        torch.save(checkpoint, path)
        if is_best:
            best_path = os.path.join(ckpt_dir, "best_model.pth")
            torch.save(checkpoint, best_path)
