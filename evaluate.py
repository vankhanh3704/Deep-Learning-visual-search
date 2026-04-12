import os
import torch
import numpy as np
import pandas as pd
import faiss
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset
from PIL import Image
from pathlib import Path
import random

# Import tu cac file local cua ban
from config import DEVICE, BATCH_SIZE, TRAIN_IMAGES_DIR
from src.models.siamese import SiameseNetwork
from src.data.dataset import load_and_clean, get_transforms

# ==========================================
# 1. KHOI TAO DATASET RIENG CHO EVALUATION
# ==========================================
class EvalDataset(Dataset):
    def __init__(self, df, transform=None):
        self.df = df.reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = Path(TRAIN_IMAGES_DIR) / row["image"]
        img = Image.open(img_path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, row["label_group"]

def evaluate_model():
    print("\n" + "="*50)
    print("BAT DAU DANH GIA MO HINH (Recall@5 & mAP@5) TREN VS CODE")
    print("="*50)

    # ==========================================
    # 2. CHUAN BI DU LIEU (Chi lay tap Validation)
    # ==========================================
    df = load_and_clean()
    SEED = 42
    random.seed(SEED); np.random.seed(SEED)
    train_idx, val_idx = [], []
    VAL_SPLIT = 0.1

    # Chia tap Validation y het nhu luc Train
    for g, grp in df.groupby("label_group"):
        idxs = grp.index.tolist()
        random.shuffle(idxs)
        n_val = int(len(idxs) * VAL_SPLIT)
        if n_val >= 2 and (len(idxs) - n_val) >= 2:
            val_idx.extend(idxs[:n_val])
        elif len(idxs) >= 4:
            val_idx.extend(idxs[:2])

    val_df = df.iloc[val_idx].reset_index(drop=True)
    print(f"So luong anh tap Validation: {len(val_df)}")

    eval_dataset = EvalDataset(val_df, transform=get_transforms("val"))
    eval_loader = DataLoader(eval_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0) # Set num_workers=0 tren Windows cho an toan

    # ==========================================
    # 3. LOAD MODEL C (DUONG DAN LOCAL VS CODE)
    # ==========================================
    model = SiameseNetwork().to(DEVICE)
    
    # SU DUNG DUONG DAN TUONG DOI TREN MAY TINH
    model_path = "./saved_models/model_c_hard_mining.pth" 
    
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=DEVICE))
        print(f"Da nap trong so Model C tu: {model_path}")
    else:
        print(f"LOI: Khong tim thay {model_path}. Hay chac chan ban da tai file pth vao thu muc saved_models!")
        return

    model.eval()

    # ==========================================
    # 4. TRICH XUAT VECTOR (EMBEDDINGS)
    # ==========================================
    print("Dang trich xuat Vector dac trung...")
    all_embeddings = []
    all_labels = []

    with torch.no_grad():
        for images, labels in tqdm(eval_loader, desc="Extracting"):
            images = images.to(DEVICE)
            embeddings = model.get_embedding(images)
            all_embeddings.append(embeddings.cpu().numpy())
            all_labels.extend(labels)

    all_embeddings = np.vstack(all_embeddings)
    all_labels = np.array(all_labels)

    # ==========================================
    # 5. DUNG FAISS TIM KIEM VA TINH TOAN METRICS
    # ==========================================
    print("Dang xay dung kho FAISS tam thoi de danh gia...")
    d = all_embeddings.shape[1]
    index = faiss.IndexFlatL2(d)
    index.add(all_embeddings)

    K = 6 
    print(f"Dang tim kiem Top 5 ket qua gan nhat cho moi anh...")
    distances, indices = index.search(all_embeddings, K)

    recall_at_5_count = 0
    map_at_5_sum = 0.0

    for i in tqdm(range(len(all_labels)), desc="Calculating Metrics"):
        query_label = all_labels[i]
        
        retrieved_indices = indices[i][1:K] 
        retrieved_labels = all_labels[retrieved_indices]
        
        is_hit = query_label in retrieved_labels
        if is_hit:
            recall_at_5_count += 1
            
        hits = 0
        precision_sum = 0.0
        for rank, label in enumerate(retrieved_labels):
            if label == query_label:
                hits += 1
                precision_sum += hits / (rank + 1.0)
                
        total_positives_in_val = min(5, (all_labels == query_label).sum() - 1)
        if total_positives_in_val > 0:
            ap_5 = precision_sum / total_positives_in_val
            map_at_5_sum += ap_5

    # ==========================================
    # 6. IN KET QUA
    # ==========================================
    recall_at_5 = (recall_at_5_count / len(all_labels)) * 100
    map_at_5 = (map_at_5_sum / len(all_labels)) * 100

    print("\n" + "="*50)
    print("KET QUA DANH GIA MO HINH (EVALUATION METRICS):")
    print(f"-> Recall@5 : {recall_at_5:.2f}%")
    print(f"-> mAP@5    : {map_at_5:.2f}%")
    print("="*50)

if __name__ == "__main__":
    evaluate_model()