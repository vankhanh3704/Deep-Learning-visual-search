import os
import sys
import importlib

# Ep Python nhan dien lai thu muc vua duoc tao tu O Code 1
working_code_dir = "/kaggle/working/my_project"
os.chdir(working_code_dir)
if working_code_dir not in sys.path:
    sys.path.insert(0, working_code_dir)

importlib.invalidate_caches()

# Xoa cac module cu kẹt trong bo nho
modules_to_remove = [mod for mod in sys.modules if mod == 'config' or mod.startswith('src')]
for mod in modules_to_remove:
    del sys.modules[mod]

# ========================================================
# BAT DAU IMPORT VA HUAN LUYEN MODEL C
# ========================================================
import torch
import torch.optim as optim
from config import LEARNING_RATE, NUM_EPOCHS, DEVICE
from src.models.siamese import SiameseNetwork
from src.models.loss import TripletLoss
from src.data.dataset import get_dataloader 

def train_model_c():
    print("\n" + "="*50)
    print(f"BAT DAU HUAN LUYEN MODEL C (HARD NEGATIVE) TREN {DEVICE}")
    print("="*50)

    train_loader = get_dataloader(split="train", hard_negative=True)
    val_loader = get_dataloader(split="val", hard_negative=False) 
    print(f"So batch tap Train: {len(train_loader)} | So batch tap Val: {len(val_loader)}\n")
    
    model = SiameseNetwork().to(DEVICE)
    
    # Tu dong quet tim trong so Model B de nap vao
    pretrained_path = None
    for root, dirs, files in os.walk("/kaggle/input"):
        if "best_model_b.pth" in files:
            pretrained_path = os.path.join(root, "best_model_b.pth")
            break
        elif "best_model.pth" in files:
            pretrained_path = os.path.join(root, "best_model.pth")
            
    if pretrained_path:
        print(f"Da tim thay trong so Model cu tai: {pretrained_path}")
        model.load_state_dict(torch.load(pretrained_path))
        print("Da nap trong so thanh cong. Bat dau Fine-tuning chuyen sau!")
    else:
        print("Canh bao: Khong tim thay file best_model.pth. Model hoc lai tu dau.")
    
    criterion = TripletLoss(margin=1.0)
    optimizer = optim.Adam(model.parameters(), lr=1e-5) 

    save_dir = "/kaggle/working/saved_models"
    os.makedirs(save_dir, exist_ok=True)
    best_model_c_path = f"{save_dir}/model_c_hard_mining.pth"
    
    best_val_loss = float('inf') 

    for epoch in range(1, NUM_EPOCHS + 1):
        # ----------------------------------------------------
        # TAP TRAIN (CO HARD NEGATIVE)
        # ----------------------------------------------------
        model.train()
        running_loss = 0.0

        for batch_idx, (anchor, positive, negative) in enumerate(train_loader):
            anchor, positive, negative = anchor.to(DEVICE), positive.to(DEVICE), negative.to(DEVICE)
            optimizer.zero_grad()

            # XU LY TENSOR 5D CUA HARD NEGATIVE MINING
            if negative.dim() == 5:
                b, n_hard, c, h, w = negative.shape
                # Phang hoa negative thanh mang 4D (batch * n_hard)
                negative_flat = negative.view(b * n_hard, c, h, w)

                # Chay lay vector dac trung (embedding) truoc de tiet kiem RAM GPU
                anchor_emb = model.get_embedding(anchor)       
                positive_emb = model.get_embedding(positive)   
                negative_emb = model.get_embedding(negative_flat) 

                # Nhan ban Anchor va Positive len 5 lan de ghep cap voi 5 Negative
                anchor_emb = anchor_emb.repeat_interleave(n_hard, dim=0)
                positive_emb = positive_emb.repeat_interleave(n_hard, dim=0)
            else:
                # Neu tat hard negative thi chay nhu binh thuong
                anchor_emb, positive_emb, negative_emb = model(anchor, positive, negative)

            loss = criterion(anchor_emb, positive_emb, negative_emb)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

            if batch_idx % 50 == 0:
                print(f"  [Train] Epoch {epoch}/{NUM_EPOCHS} | Batch {batch_idx}/{len(train_loader)} | Loss: {loss.item():.4f}")

        avg_train_loss = running_loss / len(train_loader)

        # ----------------------------------------------------
        # TAP VAL (KHONG CO HARD NEGATIVE, DE NHU BINH THUONG)
        # ----------------------------------------------------
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for anchor, positive, negative in val_loader:
                anchor, positive, negative = anchor.to(DEVICE), positive.to(DEVICE), negative.to(DEVICE)
                anchor_emb, positive_emb, negative_emb = model(anchor, positive, negative)
                loss = criterion(anchor_emb, positive_emb, negative_emb)
                val_loss += loss.item()
                
        avg_val_loss = val_loss / len(val_loader)
        
        print(f"TONG KET EPOCH {epoch} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), best_model_c_path)
            print(f"   Ky luc moi! Da luu file {best_model_c_path}\n")
        else:
            print("   Val Loss khong giam.\n")

    print(f"HOAN TAT MODEL C! Best model (Val Loss = {best_val_loss:.4f}) nam tai: {best_model_c_path}")

if __name__ == "__main__":
    train_model_c()