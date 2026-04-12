import glob
import faiss
import numpy as np
import pickle
import torch
import os
from torchvision import transforms
from PIL import Image

from config import DEVICE, IMAGE_SIZE, TRAIN_IMAGES_DIR, FAISS_INDEX_PATH
from src.models.siamese import SiameseNetwork
from backend.faiss_search import FaissSearchEngine

print("Dang khoi dong qua trinh xay dung FAISS Index...")

# 1. Load Model
model = SiameseNetwork().to(DEVICE)
model.load_state_dict(torch.load("./saved_models/model_c_hard_mining.pth", map_location=DEVICE))
model.eval()

# 2. Khoi tao Engine
engine = FaissSearchEngine()
transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# 3. Quet lay danh sach anh tu thu muc Train de trich xuat dac trung va dua vao FAISS Index
image_list = glob.glob(f"{TRAIN_IMAGES_DIR}/*.jpg")

# 4. Trich xuat vector va dua vao FAISS
engine.build_index_from_model(model, image_list, transform)

# 5. Luu xuong dia cung de API load lai
print("Dang luu kho du lieu xuong o cung...")

os.makedirs(os.path.dirname(FAISS_INDEX_PATH), exist_ok=True)

faiss.write_index(engine.index, FAISS_INDEX_PATH)

with open(FAISS_INDEX_PATH.replace(".index", "_paths.pkl"), "wb") as f:
    pickle.dump(engine.image_paths, f)

print(f"Da luu FAISS Index thanh cong tai: {FAISS_INDEX_PATH}")