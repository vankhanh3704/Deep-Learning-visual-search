import torch.nn as nn
import torchvision.models as models
from config import BACKBONE_NAME

class SoftmaxBaseline(nn.Module):
    def __init__(self, num_classes, backbone=BACKBONE_NAME, pretrained=True):
        super().__init__()
        if backbone == "resnet18":
            base = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None)
            feat_dim = 512
        else:
            base = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None)
            feat_dim = 2048

        self.backbone = nn.Sequential(*list(base.children())[:-1])
        # Thay vì xuất ra Embedding 256 chiều, mạng Softmax xuất ra dự đoán cho từng class
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(feat_dim, num_classes)
        )

    def forward(self, x):
        feat = self.backbone(x)
        return self.classifier(feat)
        
    # Thêm hàm này để Người 3 vẫn có thể trích xuất vector embedding khi test FAISS
    def get_embedding(self, x):
        feat = self.backbone(x)
        return torch.flatten(feat, 1)