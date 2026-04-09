import os
import sys
import importlib
import random
import numpy as np

working_code_dir = "/kaggle/working/my_project"
os.chdir(working_code_dir)
if working_code_dir not in sys.path:
    sys.path.insert(0, working_code_dir)

importlib.invalidate_caches()

modules_to_remove = [mod for mod in sys.modules if mod == 'config' or mod.startswith('src')]
for mod in modules_to_remove:
    del sys.modules[mod]

# ========================================================
# IMPORT VA HUAN LUYEN MODEL A (CO VALIDATION CHUAN A+)
# ========================================================
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import pandas as pd
from pathlib import Path

from config import LEARNING_RATE, NUM_EPOCHS, DEVICE, BATCH_SIZE, TRAIN_CSV, TRAIN_IMAGES_DIR
from src.models.baseline import SoftmaxBaseline
from src.data.dataset import get_transforms, load_and_clean

class ClassificationDataset(Dataset):
    def __init__(self, df, label_map, transform=None):
        self.df = df.reset_index(drop=True)
        self.transform = transform
        self.label_map = label_map

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = Path(TRAIN_IMAGES_DIR) / row["image"]
        img = Image.open(img_path).convert("RGB")
        
        if self.transform:
            img = self.transform(img)
            
        label = torch.tensor(self.label_map[row['label_group']], dtype=torch.long)
        return img, label

def train_model_a():
    print("\n" + "="*50)
    print(f"BAT DAU HUAN LUYEN MODEL A (SOFTMAX BASELINE) TREN {DEVICE}")
    print("="*50)
    
    df = load_and_clean()
    
    # 1. Tao bo tu dien anh xa nhan sang ID de dung chung cho ca Train va Val
    unique_labels = df['label_group'].unique()
    label_map = {label: idx for idx, label in enumerate(unique_labels)}
    num_classes = len(unique_labels)
    print(f"Tong so class (label_group) de phan loai: {num_classes}")

    # 2. Chia du lieu Train / Val y het nhu Model B va C
    SEED = 42
    random.seed(SEED); np.random.seed(SEED)
    train_idx, val_idx = [], []
    VAL_SPLIT = 0.1

    for g, grp in df.groupby("label_group"):
        idxs = grp.index.tolist()
        random.shuffle(idxs)
        n_val = int(len(idxs) * VAL_SPLIT)
        
        if n_val >= 2 and (len(idxs) - n_val) >= 2:
            val_idx.extend(idxs[:n_val])
            train_idx.extend(idxs[n_val:])
        elif len(idxs) >= 4:
            val_idx.extend(idxs[:2])
            train_idx.extend(idxs[2:])
        else:
            train_idx.extend(idxs)

    train_df = df.iloc[train_idx]
    val_df = df.iloc[val_idx]

    # 3. Khoi tao Dataset va DataLoader
    train_dataset = ClassificationDataset(train_df, label_map, transform=get_transforms("train"))
    val_dataset = ClassificationDataset(val_df, label_map, transform=get_transforms("val"))
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
    
    print(f"So batch tap Train: {len(train_loader)} | So batch tap Val: {len(val_loader)}\n")
    
    model = SoftmaxBaseline(num_classes=num_classes).to(DEVICE)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    save_dir = "/kaggle/working/saved_models"
    os.makedirs(save_dir, exist_ok=True)
    best_model_a_path = f"{save_dir}/model_a_baseline.pth"
    
    best_val_loss = float('inf')

    for epoch in range(1, NUM_EPOCHS + 1):
        # ----------------------------------------------------
        # TAP TRAIN
        # ----------------------------------------------------
        model.train()
        running_loss = 0.0

        for batch_idx, (images, labels) in enumerate(train_loader):
            images, labels = images.to(DEVICE), labels.to(DEVICE)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if batch_idx % 50 == 0:
                print(f"  [Train] Epoch {epoch}/{NUM_EPOCHS} | Batch {batch_idx}/{len(train_loader)} | Loss: {loss.item():.4f}")

        avg_train_loss = running_loss / len(train_loader)

        # ----------------------------------------------------
        # TAP VALIDATION (Kiem tra nguyen ban A+)
        # ----------------------------------------------------
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(DEVICE), labels.to(DEVICE)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                
                # Tinh them do chinh xac (Accuracy)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                
        avg_val_loss = val_loss / len(val_loader)
        val_accuracy = 100 * correct / total
        
        print(f"TONG KET EPOCH {epoch} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Val Acc: {val_accuracy:.2f}%")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), best_model_a_path)
            print(f"   Ky luc moi! Da luu file {best_model_a_path}\n")
        else:
            print("   Val Loss khong giam.\n")

    print(f"HOAN TAT MODEL A! Best model nam tai: {best_model_a_path}")

if __name__ == "__main__":
    train_model_a()