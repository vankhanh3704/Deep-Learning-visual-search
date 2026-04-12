import numpy as np
import faiss
import torch
from config import EMBEDDING_DIM, DEVICE
from PIL import Image

class FaissSearchEngine:
    def __init__(self, dimension=EMBEDDING_DIM, nlist=100):
        self.dimension = dimension
        self.nlist = nlist # So luong cum (clusters) de chia nho khong gian
        self.image_paths = [] 
        
        # Khoi tao FAISS voi IndexIVFFlat thay vi IndexFlatL2
        quantizer = faiss.IndexFlatL2(dimension)
        self.index = faiss.IndexIVFFlat(quantizer, dimension, self.nlist, faiss.METRIC_L2)
        self.index.nprobe = 10 # So luong cum gan nhat se duoc quet khi tim kiem

    def build_index_from_model(self, model, image_list, transform):
        print(f"Dang trich xuat dac trung cho {len(image_list)} anh...")
        model.eval()
        embeddings = []
        
        with torch.no_grad():
            for img_path in image_list:
                img = Image.open(img_path).convert("RGB")
                tensor_img = transform(img).unsqueeze(0).to(DEVICE)
                
                emb = model.get_embedding(tensor_img)
                embeddings.append(emb.cpu().numpy()[0])
                self.image_paths.append(img_path)
                
        embeddings_np = np.array(embeddings).astype('float32')
        faiss.normalize_L2(embeddings_np) 
        
        # Voi IVFFlat, BAT BUOC phai train index truoc khi add du lieu
        print("Dang huan luyen FAISS Index (Phan cum du lieu)...")
        self.index.train(embeddings_np)
        
        self.index.add(embeddings_np)
        print(f"Da xay dung xong FAISS Index. Tong so vector: {self.index.ntotal}")

    def search(self, query_vector, top_k=5):
        query_vector = np.array([query_vector]).astype('float32')
        faiss.normalize_L2(query_vector)
        
        distances, indices = self.index.search(query_vector, top_k)
        
        results = []
        for i, idx in enumerate(indices[0]):
            if idx != -1:
                results.append({
                    "image_path": self.image_paths[idx],
                    "distance": float(distances[0][i])
                })
        return results