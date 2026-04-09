import os
import sys
import shutil

print("DANG CHUAN BI MOI TRUONG KAGGLE (BAN CHINH CHU 100%)...")

# ========================================================
# BUOC 1: TIM DATA VA CODE
# ========================================================
input_code_dir = None
kaggle_data_path = None

for root, dirs, files in os.walk("/kaggle/input"):
    if "config.py" in files and "src" in dirs:
        input_code_dir = root
    if "train.csv" in files and "train_images" in dirs:
        kaggle_data_path = root

if not input_code_dir or not kaggle_data_path:
    print("LOI: Khong tim thay Code hoac Data. Kiem tra lai!")
    sys.exit()

# ========================================================
# BUOC 2: COPY CODE VA DON DEP CACHE
# ========================================================
working_code_dir = "/kaggle/working/my_project"

if os.path.exists(working_code_dir):
    shutil.rmtree(working_code_dir)
    
shutil.copytree(input_code_dir, working_code_dir)
os.chdir(working_code_dir)
if working_code_dir not in sys.path:
    sys.path.insert(0, working_code_dir)

modules_to_remove = [mod for mod in sys.modules if mod == 'config' or mod.startswith('src')]
for mod in modules_to_remove:
    del sys.modules[mod]

# ========================================================
# BUOC 3: SUA DUONG DAN CONFIG.PY
# ========================================================
with open("config.py", "r", encoding="utf-8") as f:
    config_text = f.read()

config_text = config_text.replace('./data/shopee-product-matching', kaggle_data_path)
config_text = config_text.replace('./data', kaggle_data_path)

with open("config.py", "w", encoding="utf-8") as f:
    f.write(config_text)

# ========================================================
# BUOC 3.5: PHAU THUAT LAI THUAT TOAN CHIA DU LIEU (CHUAN A+)
# ========================================================
with open("src/data/dataset.py", "r", encoding="utf-8") as f:
    dataset_content = f.read()

# Cat bo phan get_dataloader cu bi loi logic
dataset_content = dataset_content.split('def get_dataloader')[0]

# Thay the bang thuat toan chia Data chuan chinh
new_dataloader_code = """def get_dataloader(split="train", hard_negative=False):
    random.seed(SEED); np.random.seed(SEED)
    df = load_and_clean()

    train_idx, val_idx = [], []
    for g, grp in df.groupby("label_group"):
        idxs = grp.index.tolist()
        random.shuffle(idxs)

        # THUAT TOAN CHIA DU LIEU DAM BAO KHONG BI LOI INDEX
        n_val = int(len(idxs) * VAL_SPLIT)
        
        if n_val >= 2 and (len(idxs) - n_val) >= 2:
            # Nhom du lon -> Chia dung ty le
            val_idx.extend(idxs[:n_val])
            train_idx.extend(idxs[n_val:])
        elif len(idxs) >= 4:
            # Nhom co 4, 5 anh -> Ep cung lay 2 cho Val, con lai cho Train
            val_idx.extend(idxs[:2])
            train_idx.extend(idxs[2:])
        else:
            # Nhom co <=3 anh -> Dua 100% vao Train de khong bi thieu anh ghep cap
            train_idx.extend(idxs)

    df_split = df.iloc[train_idx if split == "train" else val_idx].reset_index(drop=True)
    ds = ShopeeDataset(df_split, transform=get_transforms(split), hard_negative=hard_negative)
    
    return DataLoader(ds, batch_size=BATCH_SIZE,
                      shuffle=(split == "train"),
                      num_workers=NUM_WORKERS,
                      pin_memory=True,
                      drop_last=(split == "train"))
"""

with open("src/data/dataset.py", "w", encoding="utf-8") as f:
    f.write(dataset_content + new_dataloader_code)

print("Da thay mau thuat toan chia Validation thanh cong!")
print("Chuan bi import model...")

# ========================================================
# BUOC 4: IMPORT VA HUAN LUYEN
# ========================================================
import torch
import torch.optim as optim
from config import LEARNING_RATE, NUM_EPOCHS, DEVICE
from src.models.siamese import SiameseNetwork
from src.models.loss import TripletLoss
from src.data.dataset import get_dataloader 

def train_model():
    print("\n" + "="*50)
    print(f"Bat dau huan luyen tren thiet bi: {DEVICE}")
    print("="*50)

    train_loader = get_dataloader(split="train")
    val_loader = get_dataloader(split="val") 
    print(f"So batch tap Train: {len(train_loader)} | So batch tap Val: {len(val_loader)}\n")
    
    model = SiameseNetwork().to(DEVICE)
    criterion = TripletLoss(margin=1.0)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    save_dir = "/kaggle/working/saved_models"
    os.makedirs(save_dir, exist_ok=True)
    best_model_path = f"{save_dir}/best_model.pth"
    
    best_val_loss = float('inf') 

    for epoch in range(1, NUM_EPOCHS + 1):
        # ----------------------------------------
        # PHASE 1: TRAINING
        # ----------------------------------------
        model.train()
        running_loss = 0.0

        for batch_idx, (anchor, positive, negative) in enumerate(train_loader):
            anchor, positive, negative = anchor.to(DEVICE), positive.to(DEVICE), negative.to(DEVICE)

            optimizer.zero_grad()
            anchor_emb, positive_emb, negative_emb = model(anchor, positive, negative)
            loss = criterion(anchor_emb, positive_emb, negative_emb)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if batch_idx % 50 == 0:
                print(f"  [Train] Epoch {epoch}/{NUM_EPOCHS} | Batch {batch_idx}/{len(train_loader)} | Loss: {loss.item():.4f}")

        avg_train_loss = running_loss / len(train_loader)

        # ----------------------------------------
        # PHASE 2: VALIDATION
        # ----------------------------------------
        model.eval()
        val_loss = 0.0
        
        with torch.no_grad():
            for anchor, positive, negative in val_loader:
                anchor, positive, negative = anchor.to(DEVICE), positive.to(DEVICE), negative.to(DEVICE)
                
                anchor_emb, positive_emb, negative_emb = model(anchor, positive, negative)
                loss = criterion(anchor_emb, positive_emb, negative_emb)
                val_loss += loss.item()
                
        avg_val_loss = val_loss / len(val_loader)

        # ----------------------------------------
        # LUU MODEL
        # ----------------------------------------
        print(f"TONG KET EPOCH {epoch} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), best_model_path)
            print(f"   Ky luc moi! Da luu file best_model.pth (Val Loss giam)\n")
        else:
            print("   Val Loss khong giam.\n")

        if epoch == 3:
            v1_path = f"{save_dir}/model_v1.pth"
            torch.save(model.state_dict(), v1_path)
            print(f"   Da xuat file trong so som (model_v1.pth) cho UI.\n")

    print(f"HOAN TAT! Best model (Val Loss = {best_val_loss:.4f}) nam tai: {best_model_path}")

if __name__ == "__main__":
    train_model()