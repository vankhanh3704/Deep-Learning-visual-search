import torch
import torch.nn as nn
import sys
import os

# Thêm thư mục gốc vào path để gọi file config.py
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from config import MARGIN, DEVICE

class TripletLoss(nn.Module):
    """
    Hàm tính toán Triplet Loss cho mạng Siamese.
    Áp dụng công thức: L(A, P, N) = max(0, ||A - P||^2 - ||A - N||^2 + margin)
    """
    def __init__(self, margin=MARGIN):
        super(TripletLoss, self).__init__()
        self.margin = margin
        # Sử dụng hàm có sẵn của PyTorch cho khoảng cách Euclide (p=2)
        self.loss_fn = nn.TripletMarginLoss(margin=self.margin, p=2)

    def forward(self, anchor, positive, negative):
        # Tính toán loss dựa trên 3 vector embedding truyền vào
        loss = self.loss_fn(anchor, positive, negative)
        return loss

# Test nhanh hàm Loss
if __name__ == "__main__":
    criterion = TripletLoss()
    # Giả lập vector đầu ra 256 chiều
    a = torch.randn(1, 256)
    p = torch.randn(1, 256)
    n = torch.randn(1, 256)
    print(f"Giá trị Loss thử nghiệm: {criterion(a, p, n).item():.4f}")