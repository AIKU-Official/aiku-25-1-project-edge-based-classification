import argparse
import time
import json

import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
from torch.utils.data import DataLoader
from tqdm import tqdm

def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate ResNet-18 on ImageNet-1k')
    parser.add_argument('--data-dir', type=str, required=True,
                        help='Path to ImageNet validation folder (should contain “val” subfolder)')
    parser.add_argument('--batch-size', type=int, default=256,
                        help='Batch size for evaluation')
    parser.add_argument('--workers', type=int, default=8,
                        help='Number of data loading workers')
    parser.add_argument('--model-path', type=str, default=None,
                        help='Optional path to a checkpoint (.pth) to load')
    parser.add_argument('--pretrained', action='store_true',
                        help='Use torchvision pretrained weights')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to run evaluation on (cuda or cpu)')
    return parser.parse_args()

def main():
    args = parse_args()

    # Model
    model = models.resnet34(pretrained=args.pretrained) # 18 -> 34 로 바꿈
    if args.model_path:
        checkpoint = torch.load(args.model_path, map_location='cpu')
        model.load_state_dict(checkpoint['state_dict'] if 'state_dict' in checkpoint else checkpoint)
    model = model.to(args.device)
    model.eval()

    # DataLoader
    val_dir = f"{args.data_dir}/val"
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    val_dataset = datasets.ImageFolder(
        val_dir,
        transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ]))
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
    )

    # Metrics
    top1_correct = 0
    top5_correct = 0
    total_samples = 0

    with torch.no_grad():
        for images, targets in tqdm(val_loader, desc='Evaluating'):
            images = images.to(args.device, non_blocking=True)
            targets = targets.to(args.device, non_blocking=True)

            outputs = model(images)
            # Compute top-1 and top-5
            _, preds = outputs.topk(5, dim=1, largest=True, sorted=True)
            total_samples += targets.size(0)

            # top-1
            top1 = preds[:, :1].eq(targets.view(-1, 1))
            top1_correct += top1.sum().item()

            # top-5
            top5 = preds.eq(targets.view(-1, 1))
            top5_correct += top5.sum().item()

    top1_acc = 100.0 * top1_correct / total_samples
    top5_acc = 100.0 * top5_correct / total_samples

    print(f"\nResults on ImageNet-S {args.data_dir.split('/')[-1]} val set:")
    print(f"  Total images: {total_samples}")
    print(f"  Top-1 Accuracy: {top1_acc:.2f}%")
    print(f"  Top-5 Accuracy: {top5_acc:.2f}%")

if __name__ == '__main__':
    main()
