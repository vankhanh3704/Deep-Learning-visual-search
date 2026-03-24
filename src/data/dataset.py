import random
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T

from config import (IMAGE_SIZE, TRAIN_CSV, TRAIN_IMAGES_DIR,
                    BATCH_SIZE, NUM_WORKERS, VAL_SPLIT,
                    MIN_GROUP_SIZE, SEED)


def get_transforms(split="train"):
    mean = [0.485, 0.456, 0.406]
    std  = [0.229, 0.224, 0.225]
    if split == "train":
        return T.Compose([
            T.Resize((IMAGE_SIZE + 32, IMAGE_SIZE + 32)),
            T.RandomCrop(IMAGE_SIZE),
            T.RandomHorizontalFlip(p=0.5),
            T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
            T.RandomGrayscale(p=0.05),
            T.ToTensor(),
            T.Normalize(mean, std),
        ])
    return T.Compose([
        T.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        T.ToTensor(),
        T.Normalize(mean, std),
    ])


def load_and_clean(csv_path=TRAIN_CSV, img_dir=TRAIN_IMAGES_DIR,
                   min_group_size=MIN_GROUP_SIZE):
    df = pd.read_csv(csv_path)
    img_dir = Path(img_dir)
    df["filepath"] = df["image"].apply(lambda x: img_dir / x)

    exists = df["filepath"].apply(lambda p: p.exists())
    print(f"[clean] Xóa {(~exists).sum()} dòng thiếu file ảnh.")
    df = df[exists].reset_index(drop=True)

    counts   = df["label_group"].value_counts()
    valid    = counts[counts >= min_group_size].index
    n_before = len(df)
    df = df[df["label_group"].isin(valid)].reset_index(drop=True)
    print(f"[clean] {n_before:,} → {len(df):,} ảnh sau khi lọc nhóm < {min_group_size}.")
    return df


class ShopeeDataset(Dataset):
    def __init__(self, df, transform=None, hard_negative=False):
        self.df            = df.reset_index(drop=True)
        self.transform     = transform
        self.hard_negative = hard_negative

        self.group_to_indices = defaultdict(list)
        for idx, row in self.df.iterrows():
            self.group_to_indices[row["label_group"]].append(idx)
        self.groups = list(self.group_to_indices.keys())

    def _load(self, path):
        img = Image.open(path).convert("RGB")
        return self.transform(img) if self.transform else T.ToTensor()(img)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        anchor_row    = self.df.iloc[idx]
        anchor_group  = anchor_row["label_group"]
        group_indices = self.group_to_indices[anchor_group]

        pos_idx      = random.choice([i for i in group_indices if i != idx])
        other_groups = [g for g in self.groups if g != anchor_group]

        anchor_t   = self._load(anchor_row["filepath"])
        positive_t = self._load(self.df.iloc[pos_idx]["filepath"])

        if self.hard_negative:
            N_HARD   = 5
            neg_imgs = []
            for g in random.sample(other_groups, min(N_HARD, len(other_groups))):
                ni = random.choice(self.group_to_indices[g])
                neg_imgs.append(self._load(self.df.iloc[ni]["filepath"]))
            while len(neg_imgs) < N_HARD:
                neg_imgs.append(neg_imgs[-1])
            return anchor_t, positive_t, torch.stack(neg_imgs)

        neg_group  = random.choice(other_groups)
        neg_idx    = random.choice(self.group_to_indices[neg_group])
        return anchor_t, positive_t, self._load(self.df.iloc[neg_idx]["filepath"])


def get_dataloader(split="train", hard_negative=False):
    random.seed(SEED); np.random.seed(SEED)
    df = load_and_clean()

    train_idx, val_idx = [], []
    for g, grp in df.groupby("label_group"):
        idxs = grp.index.tolist(); random.shuffle(idxs)
        cut  = max(1, int(len(idxs) * (1 - VAL_SPLIT)))
        train_idx.extend(idxs[:cut]); val_idx.extend(idxs[cut:])

    df_split = df.iloc[train_idx if split == "train" else val_idx].reset_index(drop=True)
    ds = ShopeeDataset(df_split, transform=get_transforms(split),
                       hard_negative=hard_negative)
    return DataLoader(ds, batch_size=BATCH_SIZE,
                      shuffle=(split == "train"),
                      num_workers=NUM_WORKERS,
                      pin_memory=True,
                      drop_last=(split == "train"))