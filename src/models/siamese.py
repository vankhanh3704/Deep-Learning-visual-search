import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

from config import BACKBONE_NAME, EMBEDDING_DIM


class SiameseNetwork(nn.Module):
    def __init__(self, backbone=BACKBONE_NAME,
                 embed_dim=EMBEDDING_DIM, pretrained=True):
        super().__init__()

        if backbone == "resnet18":
            base     = models.resnet18(
                weights=models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None)
            feat_dim = 512
        elif backbone == "resnet50":
            base     = models.resnet50(
                weights=models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None)
            feat_dim = 2048
        else:
            raise ValueError(f"Backbone không hỗ trợ: {backbone}")

        self.backbone  = nn.Sequential(*list(base.children())[:-1])
        self.projector = nn.Sequential(
            nn.Flatten(),
            nn.Linear(feat_dim, feat_dim // 2),
            nn.BatchNorm1d(feat_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.3),
            nn.Linear(feat_dim // 2, embed_dim),
        )

    def _embed(self, x):
        feat = self.backbone(x)
        emb  = self.projector(feat)
        return F.normalize(emb, p=2, dim=1)

    def forward(self, anchor, positive, negative):
        return self._embed(anchor), self._embed(positive), self._embed(negative)

    def get_embedding(self, x):
        return self._embed(x)

    def unfreeze_backbone(self, from_layer=6):
        children = list(self.backbone.children())
        for i, child in enumerate(children):
            for p in child.parameters():
                p.requires_grad = (i >= from_layer)
        print(f"[model] Freeze 0-{from_layer-1} | Unfreeze {from_layer}-{len(children)-1}")