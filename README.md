# E-commerce Visual Search System
**Siamese Network + Triplet Loss trên Shopee Product Matching Dataset**

Nhóm: 13

---

## Cấu trúc thư mục

```
ecommerce_visual_search/
├── .gitignore            ← Chặn data/, saved_models/, *.pth
├── config.py             ← ⚠️ NGUỒN CHÂN LÝ — đọc trước khi code
├── requirements.txt
├── README.md
│
├── data/                 ← KHÔNG push lên Git
│   ├── train.csv
│   ├── test.csv
│   └── train_images/
│
├── notebooks/
│   └── eda_shopee.ipynb  ← Người 1: Phân tích phân bổ label_group
│
├── src/
│   ├── data/            
│   │   ├── __init__.py
│   │   ├── dataset.py    ← ShopeeDataset: sinh (Anchor, Positive, Negative)
│   │   └── augmentation.py
│   │
│   ├── models/          
│   │   ├── __init__.py
│   │   ├── siamese.py    ← SiameseNetwork + ResNet backbone
│   │   └── loss.py       ← TripletLoss + Batch Hard
│   │
│   ├── training/        
│   │   ├── __init__.py
│   │   ├── train.py      ← Vòng lặp huấn luyện chính
│   │   ├── train_dummy.py← Chạy thử 100 ảnh, kiểm tra tensor shapes
│   │   └── hard_mining.py← Hard Negative Mining
│   │
│   └── evaluation/      
│       ├── __init__.py
│       ├── evaluate.py
│       └── metrics.py    ← mAP@K, Recall@K, Precision@K
│
├── backend/             
│   ├── api.py            ← FastAPI: endpoint /search/
│   └── faiss_search.py   ← IndexFlatL2 → IndexIVFFlat
│
├── frontend/            
│   └── app.py            ← Streamlit UI
│
└── saved_models/      
    ├── model_v1.pth
    └── best_model.pth
```

---

## Bắt đầu nhanh

```bash
# 1. Clone repo và cài thư viện
git clone <repo_url>
cd ecommerce_visual_search
pip install -r requirements.txt

# 2. Tải dữ liệu Kaggle (cần kaggle.json tại ~/.kaggle/)
kaggle competitions download -c shopee-product-matching -p data/

# 3. Kiểm tra pipeline (không cần GPU, không cần data)
python -m src.training.train_dummy

# 4. Train thật
python -m src.training.train

# 5. Build FAISS index
python -m backend.faiss_search build

# 6. Khởi động API
uvicorn backend.api:app --host 0.0.0.0 --port 8000 --reload

# 7. Mở giao diện
streamlit run frontend/app.py
```
