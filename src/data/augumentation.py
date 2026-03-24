"""
src/data/augmentation.py — Người 1 (Văn)
Tách riêng các pipeline augmentation để dễ thay đổi mà không đụng dataset.py

Cách dùng:
    from src.data.augmentation import get_transforms
    transform = get_transforms("train")
"""

import torchvision.transforms as T
from config import IMAGE_SIZE


def get_transforms(split: str = "train"):
    """
    Trả về transform pipeline tương ứng với split.

    split='train' → augmentation mạnh:
        - Resize lên 256 rồi RandomCrop về 224 (tránh mất viền)
        - RandomHorizontalFlip: lật ngang ngẫu nhiên
        - ColorJitter: thay đổi độ sáng, tương phản, màu sắc
        - RandomGrayscale: ngẫu nhiên chuyển sang ảnh xám (5%)

    split='val' / 'test' → chỉ resize + normalize, không augment
        (đảm bảo đánh giá nhất quán trên ảnh gốc)
    """
    # ImageNet mean & std — dùng vì backbone ResNet pre-trained trên ImageNet
    mean = [0.485, 0.456, 0.406]
    std  = [0.229, 0.224, 0.225]

    if split == "train":
        return T.Compose([
            T.Resize((IMAGE_SIZE + 32, IMAGE_SIZE + 32)),  # 256×256
            T.RandomCrop(IMAGE_SIZE),                      # cắt ngẫu nhiên → 224×224
            T.RandomHorizontalFlip(p=0.5),
            T.ColorJitter(
                brightness=0.3,
                contrast=0.3,
                saturation=0.3,
                hue=0.1,
            ),
            T.RandomGrayscale(p=0.05),
            T.ToTensor(),
            T.Normalize(mean, std),
        ])

    # val / test
    return T.Compose([
        T.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        T.ToTensor(),
        T.Normalize(mean, std),
    ])


def get_strong_transforms():
    """
    Augmentation mạnh hơn — dùng thử ở Tuần 3-4 nếu model underfit.
    Thêm RandomRotation, RandomErasing so với get_transforms('train').
    """
    mean = [0.485, 0.456, 0.406]
    std  = [0.229, 0.224, 0.225]
    return T.Compose([
        T.Resize((IMAGE_SIZE + 32, IMAGE_SIZE + 32)),
        T.RandomCrop(IMAGE_SIZE),
        T.RandomHorizontalFlip(p=0.5),
        T.RandomRotation(degrees=15),
        T.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.15),
        T.RandomGrayscale(p=0.1),
        T.ToTensor(),
        T.Normalize(mean, std),
        T.RandomErasing(p=0.2, scale=(0.02, 0.2)),  # che ngẫu nhiên 1 vùng nhỏ
    ])