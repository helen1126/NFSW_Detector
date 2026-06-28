"""Abusive 类别偏置修复单元测试。

覆盖计划 fix-abusive-category-bias.md 的 V2 验证项：
- get_prompt_text 支持 text_prompts 描述性短语
- get_batch_label 基于键位置索引（解耦 prompt 字符串内容）
- CLASM_dasmil_weighted 多分类 focal loss 分支生效

注：logit_scale clamp、text_feature_regularizer、inference 端 padding 过滤等变更
属于模型/推理层，通过 V3 代码审查验证，不在此处单测。
"""
import torch
import pytest

from utils.tools import get_prompt_text, get_batch_label
from engine.train import CLASM_dasmil_weighted


# ---- 测试用配置 ----

LABEL_MAP_8 = {
    'normal': 'normal',
    'smoke': 'smoke',
    'blood': 'blood',
    'violent': 'violent',
    'abusive': 'abusive',
    'sexy': 'sexy',
    'money': 'money',
    'policy': 'policy',
}

TEXT_PROMPTS = {
    'smoke': ['a person smoking a cigarette', 'someone holding and smoking tobacco'],
    'blood': ['blood on the ground, bloody scene, gore', 'a person bleeding from an injury'],
    'violent': ['people fighting and hitting each other', 'physical altercation and violence'],
    'abusive': ['person making aggressive gestures', 'verbal harassment and threatening behavior'],
    'sexy': ['sexually suggestive content and exposure', 'inappropriate revealing clothing or acts'],
    'money': ['displaying large amounts of cash suspiciously', 'gambling or scam related content'],
    'policy': ['politically sensitive content and symbols', 'unauthorized political commentary'],
}


# ---- get_prompt_text 测试 ----

def test_get_prompt_text_with_text_prompts():
    """提供 text_prompts 时，异常类应返回描述性短语，normal 回退到裸词。"""
    prompts = get_prompt_text(LABEL_MAP_8, TEXT_PROMPTS)
    assert prompts[0] == 'normal', "normal 不在 text_prompts 中，应回退到裸词"
    assert prompts[1] == 'a person smoking a cigarette', "smoke 应使用第一个描述性短语"
    assert prompts[4] == 'person making aggressive gestures', "abusive 应使用第一个描述性短语"
    assert prompts[7] == 'politically sensitive content and symbols', "policy 应使用第一个描述性短语"


def test_get_prompt_text_without_text_prompts():
    """不提供 text_prompts 时，行为与原版一致（返回 label_map values）。"""
    prompts = get_prompt_text(LABEL_MAP_8)
    assert prompts == list(LABEL_MAP_8.values()), "无 text_prompts 时应返回 label_map values"
    # 显式传 None 也应等价
    prompts_none = get_prompt_text(LABEL_MAP_8, None)
    assert prompts_none == prompts


def test_get_prompt_text_preserves_order_and_count():
    """返回列表长度应等于 label_map 长度，顺序与 label_map 插入顺序一致。"""
    prompts = get_prompt_text(LABEL_MAP_8, TEXT_PROMPTS)
    assert len(prompts) == len(LABEL_MAP_8), "长度应与 label_map 一致（8 项）"
    keys = list(LABEL_MAP_8.keys())
    for i, key in enumerate(keys):
        if key in TEXT_PROMPTS:
            assert prompts[i] == TEXT_PROMPTS[key][0]
        else:
            assert prompts[i] == LABEL_MAP_8[key]


# ---- get_batch_label 测试 ----

def test_get_batch_label_key_based_indexing():
    """one-hot 位置应基于 label_map 键位置，而非 prompt 字符串内容查找。"""
    # 使用描述性短语作为 prompt_text（模拟 text_prompts 启用后的场景）
    prompt_text = get_prompt_text(LABEL_MAP_8, TEXT_PROMPTS)
    texts = ['smoke', 'abusive', 'normal']
    labels = get_batch_label(texts, prompt_text, LABEL_MAP_8)

    assert labels.shape == (3, 8), "应输出 [batch, num_classes] 形状"
    keys = list(LABEL_MAP_8.keys())
    # smoke 在 keys 中的位置应为 1
    assert labels[0, keys.index('smoke')] == 1.0, "smoke 的 one-hot 应在键位置 1"
    # abusive 在 keys 中的位置应为 4
    assert labels[1, keys.index('abusive')] == 1.0, "abusive 的 one-hot 应在键位置 4"
    # normal 在 keys 中的位置应为 0
    assert labels[2, keys.index('normal')] == 1.0, "normal 的 one-hot 应在键位置 0"
    # 其余位置应为 0
    assert labels[0].sum() == 1.0, "单标签每行只应有一个 1"


def test_get_batch_label_multilabel_branch():
    """len(label_map)==7 的多标签分支：hyphenated 文本应正确拆分并设置多个 one-hot。"""
    # 构造 7 类 label_map（不含 normal），触发多标签分支
    label_map_7 = {k: v for k, v in LABEL_MAP_8.items() if k != 'normal'}
    assert len(label_map_7) == 7
    prompt_text = list(label_map_7.values())
    texts = ['smoke-blood', 'abusive-sexy']
    labels = get_batch_label(texts, prompt_text, label_map_7)

    keys = list(label_map_7.keys())
    assert labels.shape == (2, 7)
    assert labels[0, keys.index('smoke')] == 1.0
    assert labels[0, keys.index('blood')] == 1.0
    assert labels[0].sum() == 2.0, "smoke-blood 应有两个 1"
    assert labels[1, keys.index('abusive')] == 1.0
    assert labels[1, keys.index('sexy')] == 1.0


# ---- CLASM_dasmil_weighted focal loss 测试 ----

def _make_clasm_inputs(B=2, T=8, C=8, device='cpu'):
    """构造 CLASM_dasmil_weighted 的最小输入。"""
    torch.manual_seed(42)
    logits = torch.randn(B, T, C, device=device, requires_grad=True)
    # labels: [B, C] one-hot
    labels = torch.zeros(B, C, device=device)
    labels[0, 0] = 1.0  # normal
    labels[1, 4] = 1.0  # abusive
    lengths = torch.tensor([T, T], device=device)
    shot_slices = [[(0, T)], [(0, T)]]
    shot_pi_list = [None, None]
    return logits, labels, lengths, shot_slices, shot_pi_list


def test_clasm_focal_loss_runs():
    """focal=True 时 CLASM_dasmil_weighted 应返回有限标量。"""
    logits, labels, lengths, shot_slices, shot_pi_list = _make_clasm_inputs()
    loss = CLASM_dasmil_weighted(
        logits, labels, lengths, shot_slices, shot_pi_list,
        device='cpu', focal=True, focal_alpha=0.25, focal_gamma=2.0,
    )
    assert loss.dim() == 0, "loss 应为标量"
    assert torch.isfinite(loss), "loss 应为有限值"
    assert loss.item() >= 0, "loss 应非负"


def test_clasm_focal_loss_gradient():
    """focal 路径梯度应能正常回传到 logits。"""
    logits, labels, lengths, shot_slices, shot_pi_list = _make_clasm_inputs()
    loss = CLASM_dasmil_weighted(
        logits, labels, lengths, shot_slices, shot_pi_list,
        device='cpu', focal=True, focal_alpha=0.25, focal_gamma=2.0,
    )
    loss.backward()
    assert logits.grad is not None, "梯度应回传到 logits"
    assert torch.isfinite(logits.grad).all(), "梯度应全部为有限值"
    assert logits.grad.abs().sum() > 0, "梯度应非全零"
