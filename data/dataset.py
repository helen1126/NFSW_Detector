import numpy as np
import torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from utils.tools import process_feat, process_split


class SVADataset(Dataset):
    def __init__(self, num_segments, csv_path, test_mode=False, label_map=None, normal=False):
        self.num_segments = num_segments
        self.test_mode = test_mode
        self.label_map = label_map
        self.normal = normal
        self.df = pd.read_csv(csv_path)
        if not self.test_mode:
            if normal:
                self.df = self.df[self.df["label"] == "normal"].reset_index(drop=True)
            else:
                self.df = self.df[self.df["label"] != "normal"].reset_index(drop=True)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        clip_feature = np.load(self.df.loc[index]["path"])
        if not self.test_mode:
            clip_feature, clip_length = process_feat(clip_feature, self.num_segments)
        else:
            clip_feature, clip_length = process_split(clip_feature, self.num_segments)
        clip_feature = torch.tensor(clip_feature, dtype=torch.float32)
        clip_label = self.df.loc[index]["label"]
        return clip_feature, clip_label, clip_length


def create_dataloaders(config):
    num_segments = config["model"]["visual_length"]
    train_csv = config["data"]["train_csv"]
    test_csv = config["data"]["test_csv"]
    label_map = config.get("label_map", None)
    batch_size = config["training"]["batch_size"]

    cuda_config = config.get("cuda", {})
    num_workers = cuda_config.get("num_workers", 4)
    pin_memory = cuda_config.get("pin_memory", True)

    normal_dataset = SVADataset(
        num_segments=num_segments,
        csv_path=train_csv,
        test_mode=False,
        label_map=label_map,
        normal=True,
    )
    anomaly_dataset = SVADataset(
        num_segments=num_segments,
        csv_path=train_csv,
        test_mode=False,
        label_map=label_map,
        normal=False,
    )
    test_dataset = SVADataset(
        num_segments=num_segments,
        csv_path=test_csv,
        test_mode=True,
        label_map=label_map,
    )

    train_loader_kwargs = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": pin_memory,
        "drop_last": True,
    }
    if num_workers > 0:
        train_loader_kwargs["persistent_workers"] = True
        train_loader_kwargs["prefetch_factor"] = 2

    normal_loader = DataLoader(normal_dataset, shuffle=True, **train_loader_kwargs)
    anomaly_loader = DataLoader(anomaly_dataset, shuffle=True, **train_loader_kwargs)

    test_kwargs = {
        "batch_size": 1,
        "shuffle": False,
        "num_workers": num_workers,
        "pin_memory": pin_memory,
    }
    if num_workers > 0:
        test_kwargs["persistent_workers"] = True
        test_kwargs["prefetch_factor"] = 2
    test_loader = DataLoader(test_dataset, **test_kwargs)

    return normal_loader, anomaly_loader, test_loader
