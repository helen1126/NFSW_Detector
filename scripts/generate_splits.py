import os
import csv
import random

random.seed(42)

features_dir = "data/features/clip_features"
splits_dir = "data/splits"
os.makedirs(splits_dir, exist_ok=True)

all_entries = []
categories = sorted(os.listdir(features_dir))

for category in categories:
    cat_dir = os.path.join(features_dir, category)
    if not os.path.isdir(cat_dir):
        continue
    for fname in os.listdir(cat_dir):
        if fname.endswith(".npy"):
            fpath = os.path.join(cat_dir, fname).replace("\\", "/")
            all_entries.append((fpath, category))

random.shuffle(all_entries)

split_idx = int(len(all_entries) * 0.8)
train_entries = all_entries[:split_idx]
test_entries = all_entries[split_idx:]

train_csv = os.path.join(splits_dir, "train.csv")
test_csv = os.path.join(splits_dir, "test.csv")

with open(train_csv, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["path", "label"])
    for path, label in train_entries:
        writer.writerow([path, label])

with open(test_csv, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["path", "label"])
    for path, label in test_entries:
        writer.writerow([path, label])

print(f"Total: {len(all_entries)} | Train: {len(train_entries)} | Test: {len(test_entries)}")
print(f"Train CSV: {train_csv}")
print(f"Test CSV: {test_csv}")

from collections import Counter
train_dist = Counter(l for _, l in train_entries)
test_dist = Counter(l for _, l in test_entries)
print(f"Train distribution: {dict(train_dist)}")
print(f"Test distribution: {dict(test_dist)}")
