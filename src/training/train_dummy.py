import torch
import torch.optim as optim
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from src.models.loss import TripletLoss
from config import BATCH_SIZE, EMBEDDING_DIM, LEARNING_RATE, DEVICE

def run_dummy_training():
    print(f"Bắt đầu huấn luyện thử nghiệm (Dummy Training) trên {DEVICE}...")
    
    # Khởi tạo hàm loss
    criterion = TripletLoss().to(DEVICE)
    
    # Tạo một "mô hình" giả (chỉ là một lớp Linear đơn giản để có tham số huấn luyện)
    dummy_model = torch.nn.Linear(EMBEDDING_DIM, EMBEDDING_DIM).to(DEVICE)
    
    # Cài đặt thuật toán tối ưu Adam
    optimizer = optim.Adam(dummy_model.parameters(), lr=LEARNING_RATE)
    
    # Mô phỏng huấn luyện với 10 "batch" dữ liệu giả
    num_dummy_batches = 10
    
    for batch_idx in range(num_dummy_batches):
        # 1. Giả lập dữ liệu sinh ra từ Siamese Network (Vector 256 chiều)
        anchor_out = torch.randn(BATCH_SIZE, EMBEDDING_DIM, requires_grad=True).to(DEVICE)
        positive_out = torch.randn(BATCH_SIZE, EMBEDDING_DIM, requires_grad=True).to(DEVICE)
        negative_out = torch.randn(BATCH_SIZE, EMBEDDING_DIM, requires_grad=True).to(DEVICE)
        
        # Cho qua dummy_model để mô phỏng tính toán
        a = dummy_model(anchor_out)
        p = dummy_model(positive_out)
        n = dummy_model(negative_out)

        # 2. Xóa gradient cũ
        optimizer.zero_grad()
        
        # 3. Tính Triplet Loss
        loss = criterion(a, p, n)
        
        # 4. Lan truyền ngược (Backpropagation)
        loss.backward()
        
        # 5. Cập nhật trọng số
        optimizer.step()
        
        print(f"Batch {batch_idx + 1}/{num_dummy_batches} | Loss: {loss.item():.4f} - Kích thước tensor: {a.shape}")

    print("Dummy Training hoàn tất! Không có lỗi kích thước Tensor.")

if __name__ == "__main__":
    run_dummy_training()