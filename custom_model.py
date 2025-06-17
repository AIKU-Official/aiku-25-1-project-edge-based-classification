import torch
import torch.nn as nn
import torchvision.models as models

class Dual_Model(nn.Module):
    def __init__(self, num_classes=1000, pretrained=True):
        super(Dual_Model, self).__init__()

        # 원본 이미지용 ResNet34 (freeze)
        self.backbone1 = models.resnet34(pretrained=pretrained)
        self.backbone1 = nn.Sequential(*list(self.backbone1.children())[:-1])

        # 엣지 이미지용 ResNet34 (trainable)
        self.backbone2 = models.resnet34(pretrained=pretrained)
        self.backbone2 = nn.Sequential(*list(self.backbone2.children())[:-1])

        # FC layer (512 * 2 → num_classes)
        self.fc = nn.Linear(512 * 2, num_classes)

    def forward(self, img, edge_img):
        feat1 = self.backbone1(img)      # [B, 512, 1, 1]
        feat2 = self.backbone2(edge_img) # [B, 512, 1, 1]

        feat1 = feat1.view(feat1.size(0), -1)  # [B, 512]
        feat2 = feat2.view(feat2.size(0), -1)  # [B, 512]

        concat_feat = torch.cat([feat1, feat2], dim=1)  # [B, 1024]
        out = self.fc(concat_feat)                      # [B, num_classes]
        return out

    def train_setup(self):
        # backbone1 (original image) trainable
        for param in self.backbone1.parameters():
            param.requires_grad = True

        # backbone2 (edge image) trainable
        for param in self.backbone2.parameters():
            param.requires_grad = True

        # fc layer trainable
        for param in self.fc.parameters():
            param.requires_grad = True
        
        self.train()
