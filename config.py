"""
config.py 
Ai muốn sửa file này phải nhắn vào group chat báo 2 người kia!
"""

import torch


# CẤU HÌNH ĐƯỜNG DẪN DỮ LIỆU & LƯU TRỮ

DATA_DIR           = "./data/shopee-product-matching"
TRAIN_CSV          = f"{DATA_DIR}/train.csv"
TEST_CSV           = f"{DATA_DIR}/test.csv"
TRAIN_IMAGES_DIR   = f"{DATA_DIR}/train_images"
MODEL_SAVE_PATH    = "./saved_models/best_model.pth"
FAISS_INDEX_PATH   = "./faiss_indexes/shopee_index.index"


# CẤU HÌNH KIẾN TRÚC MÔ HÌNH 
BACKBONE_NAME  = "resnet18"    # Có thể đổi thành "resnet50" sau này
IMAGE_SIZE     = 224           # Kích thước chuẩn đưa vào ResNet
EMBEDDING_DIM  = 256           # Chiều dài vector đặc trưng ở lớp cuối
MIN_GROUP_SIZE = 3             # Lọc nhóm ít hơn N ảnh khỏi training

# CẤU HÌNH HUẤN LUYỆN 
BATCH_SIZE     = 32
NUM_EPOCHS     = 20
LEARNING_RATE  = 1e-4          # Learning Rate cho Adam
MARGIN         = 1.0           # Margin cho hàm Triplet Loss
VAL_SPLIT      = 0.1
NUM_WORKERS    = 4
SEED           = 42

# Tự động nhận diện GPU
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# CẤU HÌNH TÌM KIẾM FAISS & GIAO DIỆN 
FAISS_INDEX_TYPE = "IndexFlatL2"   # Nâng cấp lên IndexIVFFlat ở Tuần 5
TOP_K_RESULTS    = 5               # Số ảnh trả về trên UI (Recall@5)
API_HOST         = "0.0.0.0"
API_PORT         = 8000
