import torch
import torch.nn as nn
import torch.nn.functional as F

LABEL_MAP = {
    0: {"en": "Smoke", "zh": "吸烟"},
    1: {"en": "Blood", "zh": "血腥"},
    2: {"en": "Violent", "zh": "暴力"},
    3: {"en": "Abusive", "zh": "辱骂"},
    4: {"en": "Sexy", "zh": "色情"},
    5: {"en": "Money", "zh": "金钱诈骗"},
    6: {"en": "Policy", "zh": "政治敏感"},
}

EN_TO_ZH = {"Smoke": "吸烟", "Blood": "血腥", "Violent": "暴力", "Abusive": "辱骂", "Sexy": "色情", "Money": "金钱诈骗", "Policy": "政治敏感"}

ZH_TO_EN = {"吸烟": "Smoke", "血腥": "Blood", "暴力": "Violent", "辱骂": "Abusive", "色情": "Sexy", "金钱诈骗": "Money", "政治敏感": "Policy"}


class MultiLabelClassifier(nn.Module):
    def __init__(self, input_dim=256, num_classes=7, dropout=0.3):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, input_dim)
        self.fc2 = nn.Linear(input_dim, num_classes)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.dropout(self.relu(self.fc1(x)))
        logits = self.fc2(x)
        probs = torch.sigmoid(logits)
        preds = (probs >= 0.5).float()
        max_prob, max_idx = probs.max(dim=-1)
        return logits, probs, preds, max_idx, max_prob


class MultiLabelLoss(nn.Module):
    def __init__(self, num_classes=7, class_weights=None, focal=False, focal_alpha=0.25, focal_gamma=2.0):
        super().__init__()
        self.num_classes = num_classes
        self.focal = focal
        self.focal_alpha = focal_alpha
        self.focal_gamma = focal_gamma
        if class_weights is not None:
            class_weights_tensor = torch.tensor(class_weights, dtype=torch.float32)
            self.register_buffer("class_weights", class_weights_tensor)
            self.bce = nn.BCEWithLogitsLoss(pos_weight=class_weights_tensor, reduction="none")
        else:
            self.class_weights = None
            self.bce = nn.BCEWithLogitsLoss(reduction="none")

    def forward(self, logits, targets):
        if self.focal:
            bce = self.bce(logits, targets)
            pt = torch.exp(-bce)
            alpha = self.focal_alpha
            gamma = self.focal_gamma
            loss = alpha * (1 - pt) ** gamma * bce
        else:
            loss = self.bce(logits, targets)
        if self.class_weights is not None:
            loss = loss * self.class_weights
        return loss.mean()


def get_label_info(label_id):
    return LABEL_MAP[label_id]


def en_to_zh(en_name):
    return EN_TO_ZH[en_name]


def zh_to_en(zh_name):
    return ZH_TO_EN[zh_name]


def get_all_labels():
    return LABEL_MAP
