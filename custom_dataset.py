from torchvision import transforms, datasets
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import os

class DualImageFolder(Dataset):
    def __init__(self, img_dir, edge_dir, transform=None):
        self.img_dataset = datasets.ImageFolder(img_dir)
        self.edge_dir = edge_dir
        self.transform = transform
        self.samples = self.img_dataset.samples
        self.classes = self.img_dataset.classes
        self.class_to_idx = self.img_dataset.class_to_idx

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]

        # edge 이미지 경로 구성
        rel_path = os.path.relpath(img_path, self.img_dataset.root)
        edge_path = os.path.join(self.edge_dir, rel_path)

        # 이미지 로드
        img = Image.open(img_path).convert("RGB")
        edge = Image.open(edge_path).convert("RGB")

        # 변환 적용
        if self.transform:
            img = self.transform(img)
            edge = self.transform(edge)

        return img, edge, label
