"""算法优化单元测试。

覆盖计划 optimize-algorithm-accuracy.md 的 V2 验证项：
- DistanceAdj.sigma 修复后参与计算
- focal loss 对难样本加权
- CLAS2 focal 分支生效
- 数据增强 is_random 随机性
- configs/default.yaml 优化值落地

注：lambda_v/lambda_t 初始化与 SVLA 构造相关（需加载 CLIP），通过 V3 代码审查验证，不在此处单测。
"""
import numpy as np
import torch
import yaml
import pytest

from models.layers import DistanceAdj
from engine.train import CLAS2_dasmil_weighted
from utils.tools import process_feat


def test_distance_adj_uses_sigma():
    """修复后 DistanceAdj.forward 输出应依赖于 self.sigma。"""
    adj = DistanceAdj()
    adj.sigma.data.fill_(1.0)
    out_a = adj.forward(batch_size=1, max_seqlen=8).clone()

    adj.sigma.data.fill_(0.1)
    out_b = adj.forward(batch_size=1, max_seqlen=8).clone()

    assert not torch.allclose(out_a, out_b), "DistanceAdj 输出应随 sigma 变化"
    # sigma 越小 → exp(sigma) 越小 → 衰减越快 → 远距离值越小
    assert out_b[0, 0, -1] < out_a[0, 0, -1], "sigma 减小应使远距离邻接值降低"


def test_focal_loss_hard_sample_weighting():
    """focal 权重应使难样本（低 pt）权重高于易样本（高 pt）。"""
    focal_alpha, focal_gamma = 0.25, 2.0
    pt_easy = torch.tensor(0.95)   # 易分样本
    pt_hard = torch.tensor(0.1)    # 难分样本
    w_easy = focal_alpha * (1 - pt_easy) ** focal_gamma
    w_hard = focal_alpha * (1 - pt_hard) ** focal_gamma
    assert w_hard > w_easy, "难样本 focal 权重应高于易样本"
    assert w_hard.item() > 0.2, "难样本权重应显著"


def test_clas2_focal_vs_plain():
    """CLAS2_dasmil_weighted 的 focal 分支应与 plain 分支不同，且对易样本降低 loss。"""
    B, T = 2, 8
    logits = torch.zeros(B, T, 1)
    # 易分样本：logits 使 sigmoid 后接近标签
    logits[0, :, 0] = 3.0   # normal 样本，标签 0
    logits[1, :, 0] = -3.0  # anomaly 样本，标签 1
    labels = torch.zeros(B, 8)
    labels[0, 0] = 1  # normal
    labels[1, 1] = 0  # 异常类（非 normal）
    lengths = torch.tensor([T, T])
    shot_slices = [[(0, T)], [(0, T)]]
    shot_pi = [torch.tensor([0.3]), torch.tensor([0.7])]

    loss_plain = CLAS2_dasmil_weighted(
        logits, labels, lengths, shot_slices, shot_pi,
        device=torch.device("cpu"), focal=False,
    )
    loss_focal = CLAS2_dasmil_weighted(
        logits, labels, lengths, shot_slices, shot_pi,
        device=torch.device("cpu"), focal=True, focal_alpha=0.25, focal_gamma=2.0,
    )
    assert loss_plain.item() > 0
    assert loss_focal.item() > 0
    # 易分样本 focal 应下调 loss
    assert loss_focal.item() < loss_plain.item(), "focal 应降低易样本 loss"


def test_process_feat_random_augmentation():
    """is_random=True 应产生随机性，is_random=False 应确定性。"""
    feat = np.random.randn(100, 32).astype(np.float32)
    length = 50

    np.random.seed(0)
    a1, _ = process_feat(feat, length, is_random=True)
    np.random.seed(1)
    a2, _ = process_feat(feat, length, is_random=True)
    assert not np.allclose(a1, a2), "is_random=True 不同种子应产生不同结果"

    b1, _ = process_feat(feat, length, is_random=False)
    b2, _ = process_feat(feat, length, is_random=False)
    assert np.allclose(b1, b2), "is_random=False 应确定性"

    # 短序列走 pad 分支，is_random 不影响
    short = np.random.randn(10, 32).astype(np.float32)
    s1, _ = process_feat(short, length, is_random=True)
    s2, _ = process_feat(short, length, is_random=False)
    assert np.allclose(s1, s2), "短序列 pad 分支不受 is_random 影响"


def test_config_optimization_values():
    """configs/default.yaml 应反映算法优化配置。"""
    with open("configs/default.yaml", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    assert cfg["model"]["visual_head"] == 4, "visual_head 应为 4"
    assert cfg["training"]["epochs"] == 20, "epochs 应为 20"
    assert cfg["training"]["scheduler_milestones"] == [10, 16], "milestones 应为 [10,16]"
    assert cfg["training"]["txtreg_weight"] == 0.3, "txtreg_weight 应为 0.3"
    assert cfg["training"]["focal"] is True, "focal 应启用"
    assert cfg["training"]["focal_alpha"] == 0.25
    assert cfg["training"]["focal_gamma"] == 2.0
    assert cfg["training"]["class_loss_alpha"] == 1.0
    assert all(m < cfg["training"]["epochs"] for m in cfg["training"]["scheduler_milestones"]), "milestones 应小于 epochs"
