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


def get_label_info(label_id):
    return LABEL_MAP[label_id]


def en_to_zh(en_name):
    return EN_TO_ZH[en_name]


def zh_to_en(zh_name):
    return ZH_TO_EN[zh_name]


def get_all_labels():
    return LABEL_MAP
