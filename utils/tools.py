import torch
import numpy as np


def get_batch_label(texts, prompt_text, label_map: dict):
    label_vectors = torch.zeros(0)
    if len(label_map) != 7:
        if len(label_map) == 2:
            for text in texts:
                label_vector = torch.zeros(2)
                if text == 'Normal':
                    label_vector[0] = 1
                else:
                    label_vector[1] = 1
                label_vector = label_vector.unsqueeze(0)
                label_vectors = torch.cat([label_vectors, label_vector], dim=0)
        else:
            keys = list(label_map.keys())
            for text in texts:
                label_vector = torch.zeros(len(prompt_text))
                if text in label_map:
                    idx = keys.index(text)
                    label_vector[idx] = 1

                label_vector = label_vector.unsqueeze(0)
                label_vectors = torch.cat([label_vectors, label_vector], dim=0)
    else:
        keys = list(label_map.keys())
        for text in texts:
            label_vector = torch.zeros(len(prompt_text))
            labels = text.split('-')
            for label in labels:
                if label in label_map:
                    idx = keys.index(label)
                    label_vector[idx] = 1

            label_vector = label_vector.unsqueeze(0)
            label_vectors = torch.cat([label_vectors, label_vector], dim=0)

    return label_vectors


def get_prompt_text(label_map: dict, text_prompts: dict = None):
    prompt_text = []
    for key, value in label_map.items():
        if text_prompts and key in text_prompts and text_prompts[key]:
            # Use the first descriptive phrase for richer CLIP encoding
            prompt_text.append(text_prompts[key][0])
        else:
            prompt_text.append(value)

    return prompt_text


def get_batch_mask(lengths, maxlen):
    batch_size = lengths.shape[0]
    mask = torch.empty(batch_size, maxlen)
    mask.fill_(0)
    for i in range(batch_size):
        if lengths[i] < maxlen:
            mask[i, lengths[i]:maxlen] = 1

    return mask.bool()


def random_extract(feat, t_max):
    r = np.random.randint(feat.shape[0] - t_max)
    return feat[r: r + t_max, :]


def uniform_extract(feat, t_max, avg: bool = True):
    new_feat = np.zeros((t_max, feat.shape[1])).astype(np.float32)
    r = np.linspace(0, len(feat), t_max + 1, dtype=np.int32)
    if avg == True:
        for i in range(t_max):
            if r[i] != r[i + 1]:
                new_feat[i, :] = np.mean(feat[r[i]:r[i + 1], :], 0)
            else:
                new_feat[i, :] = feat[r[i], :]
    else:
        r = np.linspace(0, feat.shape[0] - 1, t_max, dtype=np.uint16)
        new_feat = feat[r, :]

    return new_feat


def pad(feat, min_len):
    clip_length = feat.shape[0]
    if clip_length <= min_len:
        return np.pad(feat, ((0, min_len - clip_length), (0, 0)), mode='constant', constant_values=0)
    else:
        return feat


def process_feat(feat, length, is_random=False):
    clip_length = feat.shape[0]
    if feat.shape[0] > length:
        if is_random:
            return random_extract(feat, length), length
        else:
            return uniform_extract(feat, length), length
    else:
        return pad(feat, length), clip_length


def process_split(feat, length):
    clip_length = feat.shape[0]
    if clip_length < length:
        return pad(feat, length), clip_length
    else:
        split_num = int(clip_length / length) + 1
        for i in range(split_num):
            if i == 0:
                split_feat = feat[i * length:i * length + length, :].reshape(1, length, feat.shape[1])
            elif i < split_num - 1:
                split_feat = np.concatenate(
                    [split_feat, feat[i * length:i * length + length, :].reshape(1, length, feat.shape[1])], axis=0)
            else:
                split_feat = np.concatenate([split_feat,
                                             pad(feat[i * length:i * length + length, :], length).reshape(1, length,
                                                                                                          feat.shape[
                                                                                                              1])],
                                            axis=0)

        return split_feat, clip_length


# CLIP 变体与输出维度的映射
CLIP_VARIANT_DIMS = {
    "RN50": 1024,
    "RN101": 512,
    "RN50x4": 640,
    "ViT-B/32": 512,
    "ViT-B/16": 512,
    "ViT-L/14": 768,
    "ViT-L/14@336px": 768,
}


def validate_clip_config(model_cfg: dict):
    """校验 clip_variant 与 embed_dim/visual_width 的一致性。

    若不一致，抛出 ValueError 并给出修复建议。
    """
    variant = model_cfg.get("clip_variant", "ViT-B/16")
    embed_dim = model_cfg.get("embed_dim", 512)
    visual_width = model_cfg.get("visual_width", 512)
    expected = CLIP_VARIANT_DIMS.get(variant)
    if expected is None:
        raise ValueError(
            f"未知的 clip_variant: {variant}。"
            f"支持的变体: {list(CLIP_VARIANT_DIMS.keys())}"
        )
    if embed_dim != expected or visual_width != expected:
        raise ValueError(
            f"配置不一致: clip_variant={variant} 期望 embed_dim=visual_width={expected}, "
            f"但配置为 embed_dim={embed_dim}, visual_width={visual_width}。"
            f"请同步修改 embed_dim 和 visual_width 为 {expected}。"
        )