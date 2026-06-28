"""骨干网络配置单元测试。

覆盖计划 configurable-backbone-upgrade.md 的验证项：
- validate_clip_config 对合法配置通过
- validate_clip_config 对维度不匹配配置抛出 ValueError
- validate_clip_config 对未知变体抛出 ValueError
- CLIP_VARIANT_DIMS 映射完整且维度正确
"""
import pytest

from utils.tools import validate_clip_config, CLIP_VARIANT_DIMS


def test_validate_clip_config_valid():
    """合法配置应通过校验：ViT-B/16+512, ViT-L/14+768。"""
    # ViT-B/16 默认配置
    validate_clip_config({"clip_variant": "ViT-B/16", "embed_dim": 512, "visual_width": 512})
    # ViT-L/14 高精度配置
    validate_clip_config({"clip_variant": "ViT-L/14", "embed_dim": 768, "visual_width": 768})
    # 缺失 clip_variant 时应默认 ViT-B/16 + 512
    validate_clip_config({"embed_dim": 512, "visual_width": 512})


def test_validate_clip_config_mismatch():
    """维度不匹配应抛出 ValueError。"""
    with pytest.raises(ValueError, match="配置不一致"):
        validate_clip_config({"clip_variant": "ViT-L/14", "embed_dim": 512, "visual_width": 512})
    with pytest.raises(ValueError, match="配置不一致"):
        validate_clip_config({"clip_variant": "ViT-B/16", "embed_dim": 768, "visual_width": 768})
    # embed_dim 与 visual_width 不一致也应报错
    with pytest.raises(ValueError, match="配置不一致"):
        validate_clip_config({"clip_variant": "ViT-B/16", "embed_dim": 512, "visual_width": 768})


def test_validate_clip_config_unknown_variant():
    """未知变体应抛出 ValueError。"""
    with pytest.raises(ValueError, match="未知的 clip_variant"):
        validate_clip_config({"clip_variant": "ViT-H/14", "embed_dim": 1024, "visual_width": 1024})


def test_clip_variant_dims_mapping():
    """CLIP_VARIANT_DIMS 应包含所有支持的变体且维度正确。"""
    expected = {
        "RN50": 1024,
        "RN101": 512,
        "RN50x4": 640,
        "ViT-B/32": 512,
        "ViT-B/16": 512,
        "ViT-L/14": 768,
        "ViT-L/14@336px": 768,
    }
    assert set(CLIP_VARIANT_DIMS.keys()) == set(expected.keys()), "变体集合应匹配"
    for variant, dim in expected.items():
        assert CLIP_VARIANT_DIMS[variant] == dim, f"{variant} 维度应为 {dim}"
