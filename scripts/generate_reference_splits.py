"""从参考实现的 CSV 生成项目训练/测试划分。

参考 CSV 中的路径前缀为 /root/autodl-tmp/VadCLIP-main/clip_features/，
需重写为项目的 data/features/clip_features/。

使用方法：
    python scripts/generate_reference_splits.py
"""

import csv
import os
from collections import Counter

REFERENCE_DIR = "svla_reference/list"
PROJECT_SPLITS_DIR = "data/splits"
PATH_PREFIX_OLD = "/root/autodl-tmp/VadCLIP-main/clip_features/"
PATH_PREFIX_NEW = "data/features/clip_features/"

SPLITS = [
    ("Sva_CLIP_rgb.csv", "train.csv"),
    ("Sva_CLIP_rgbtest.csv", "test.csv"),
]


def rewrite_path(path: str) -> str:
    return path.replace(PATH_PREFIX_OLD, PATH_PREFIX_NEW)


def convert_csv(src_path: str, dst_path: str) -> int:
    os.makedirs(os.path.dirname(dst_path), exist_ok=True)
    count = 0
    labels = []
    with open(src_path, "r", encoding="utf-8") as fin, \
         open(dst_path, "w", newline="", encoding="utf-8") as fout:
        reader = csv.reader(fin)
        writer = csv.writer(fout)
        header = next(reader)
        writer.writerow(header)
        for row in reader:
            if len(row) < 2:
                continue
            row[0] = rewrite_path(row[0])
            writer.writerow(row)
            count += 1
            labels.append(row[1])
    return count, labels


def main():
    os.makedirs(PROJECT_SPLITS_DIR, exist_ok=True)

    for ref_name, proj_name in SPLITS:
        src = os.path.join(REFERENCE_DIR, ref_name)
        dst = os.path.join(PROJECT_SPLITS_DIR, proj_name)
        if not os.path.exists(src):
            print(f"ERROR: Reference CSV not found: {src}")
            continue
        count, labels = convert_csv(src, dst)
        dist = Counter(labels)
        print(f"{proj_name}: {count} entries")
        print(f"  Labels: {dict(dist)}")
        print(f"  Written to: {dst}")


if __name__ == "__main__":
    main()
