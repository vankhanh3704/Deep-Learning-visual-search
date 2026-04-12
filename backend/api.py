from fastapi import FastAPI, UploadFile, File
import uvicorn
import torch
from PIL import Image
import io
import os
import faiss
import pickle
import time

from config import DEVICE, IMAGE_SIZE, FAISS_INDEX_PATH
from src.models.siamese import SiameseNetwork
from backend.faiss_search import FaissSearchEngine
from torchvision import transforms

app = FastAPI(title="Shopee Visual Search API")

# ==========================================
# 1. KHOI TAO VA LOAD MODEL
# ==========================================
print("Dang tai Siamese Network Model...")
model = SiameseNetwork().to(DEVICE)

# Ban co the doi ten file thanh model_c_hard_mining.pth neu da train xong Model C
model_path = "./saved_models/model_c_hard_mining.pth" 

if os.path.exists(model_path):
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    print(f"Da nap trong so model tu: {model_path}")
else:
    print(f"CANH BAO: Khong tim thay {model_path}. Kiem tra lai duong dan!")
    
model.eval()

# ==========================================
# 2. KHOI TAO VA LOAD FAISS ENGINE
# ==========================================
print("Dang nap kho du lieu FAISS...")
# Khoi tao voi nlist=100 phu hop cho IndexIVFFlat cua Tuan 5-6
faiss_engine = FaissSearchEngine(nlist=100) 

try:
    faiss_engine.index = faiss.read_index(FAISS_INDEX_PATH)
    with open(FAISS_INDEX_PATH.replace(".index", "_paths.pkl"), "rb") as f:
        faiss_engine.image_paths = pickle.load(f)
    print(f"Da nap thanh cong {faiss_engine.index.ntotal} anh vao bo nho FAISS!")
except Exception as e:
    print(f"CANH BAO: Chua tim thay file FAISS. Ban can chay build_faiss.py truoc. Loi: {e}")

# ==========================================
# 3. TRANSFORM DATA
# ==========================================
# Dong bo kich thuoc voi luc Huan luyen (Train)
transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# ==========================================
# 4. ENDPOINT TIM KIEM
# ==========================================
@app.post("/search/")
async def search_image(file: UploadFile = File(...)):
    start_time = time.time() # Bat dau bam gio
    
    try:
        # Doc anh tu Frontend gui len
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        
        # Chuyen anh thanh tensor va dua vao Model de lay vector dac trung
        tensor_img = transform(image).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            query_vector = model.get_embedding(tensor_img).cpu().numpy()[0]
        
        # Dung vector do de tim kiem trong kho FAISS
        search_results = faiss_engine.search(query_vector, top_k=5)
        
        inference_time = time.time() - start_time # Chot thoi gian
        
        return {
            "status": "success",
            "inference_time": inference_time,
            "results": search_results
        }
        
    except Exception as e:
        return {"status": "error", "message": str(e)}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)