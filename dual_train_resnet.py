import argparse
import time
import copy

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
from torch.utils.data import DataLoader
from tqdm import tqdm
from custom_model import Dual_Model
from custom_dataset import DualImageFolder
import os


def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune ResNet-18 on custom data")
    parser.add_argument("--data-dir", type=str, required=True,
                        help="Root directory of data (expects 'train' and 'val' subfolders)")
    parser.add_argument("--log-dir", type=str, required=True,
                        help="Log directory")
    parser.add_argument("--edge-dir", type=str, required=True,
                        help="Root directory of edge data (expects 'train' and 'val' subfolders)")
    parser.add_argument("--batch-size", type=int, default=64,
                        help="Batch size for training and validation")
    parser.add_argument("--epochs", type=int, default=10,
                        help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=1e-3,
                        help="Initial learning rate")
    parser.add_argument("--momentum", type=float, default=0.9,
                        help="SGD momentum")
    parser.add_argument("--weight-decay", type=float, default=1e-4,
                        help="Weight decay (L2 penalty)")
    parser.add_argument("--pretrained", action="store_true",
                        help="Start from ImageNet-pretrained weights")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device to run on: 'cuda' or 'cpu'")
    parser.add_argument("--save-path", type=str, default="resnet18_finetuned.pth",
                        help="Where to save the best model checkpoint")
    return parser.parse_args()


def build_dataloaders(img_dir, edge_dir, batch_size, num_workers=4):
    train_transforms = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])
    val_transforms = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    train_dataset = DualImageFolder(
        img_dir=os.path.join(img_dir, "train"),
        edge_dir=os.path.join(edge_dir, "train"),
        transform=train_transforms
    )
    val_dataset = DualImageFolder(
        img_dir=os.path.join(img_dir, "val"),
        edge_dir=os.path.join(edge_dir, "val"),
        transform=val_transforms
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=True)

    return train_loader, val_loader, len(train_dataset.classes)


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    running_corrects = 0
    total = 0

    for imgs, edges, labels in tqdm(loader, desc="  Training", leave=False):
        imgs = imgs.to(device, non_blocking=True)
        edges = edges.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad()
        outputs = model(imgs, edges)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        preds = outputs.argmax(dim=1)
        running_loss += loss.item() * imgs.size(0)
        running_corrects += (preds == labels).sum().item()
        total += imgs.size(0)

    epoch_loss = running_loss / total
    epoch_acc  = running_corrects / total
    return epoch_loss, epoch_acc


def validate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    running_corrects = 0
    total = 0

    with torch.no_grad():
        for inputs, edges, labels in tqdm(loader, desc="  Validating", leave=False):
            inputs = inputs.to(device, non_blocking=True)
            edges = edges.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            outputs = model(inputs, edges)
            loss = criterion(outputs, labels)

            preds = outputs.argmax(dim=1)
            running_loss += loss.item() * inputs.size(0)
            running_corrects += (preds == labels).sum().item()
            total += inputs.size(0)

    epoch_loss = running_loss / total
    epoch_acc  = running_corrects / total
    return epoch_loss, epoch_acc

def log(args, msg):
    print(msg)
    with open(args.log_dir, "a+") as f:
        f.write(msg + "\n")
        f.flush()

def main():
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # Data loaders
    train_loader, val_loader, num_classes = build_dataloaders(
        args.data_dir, args.edge_dir, args.batch_size
    )

    # Model setup
    model = Dual_Model(pretrained=args.pretrained)
    
    model.train_setup()
            
    model = model.to(device)
        
    criterion = nn.CrossEntropyLoss()

    # 3) Build optimizer over just the fc params
    optimizer = optim.AdamW(
        model.fc.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
        betas=(0.9, 0.999),
        eps=1e-8
    )
    # Optional LR scheduler
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    log(args, "===== Training Start =====")
    log(args, f"Device: {device}")
    log(args, f"Batch Size: {args.batch_size}, LR: {args.lr}, Epochs: {args.epochs}")
    log(args, f"Input: {args.data_dir}, {args.edge_dir}")
    
    for epoch in range(1, args.epochs + 1):
        log(args, f"Epoch {epoch}/{args.epochs}")
        since = time.time()

        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc     = validate(model, val_loader, criterion, device)
        scheduler.step()

        elapsed = time.time() - since
        log(args, f"  Train Loss: {train_loss:.4f}  Acc: {train_acc:.4f}")
        log(args, f"  Val   Loss: {val_loss:.4f}  Acc: {val_acc:.4f}")
        log(args, f"  Time: {elapsed//60:.0f}m {elapsed%60:.0f}s")

        # deep copy the best model
        if val_acc > best_acc:
            best_acc = val_acc
            best_model_wts = copy.deepcopy(model.state_dict())

    print(f"Best Validation Acc: {best_acc:.4f}")
    # load best weights and save
    model.load_state_dict(best_model_wts)
    torch.save(model.state_dict(), args.save_path)
    print(f"Saved best model to {args.save_path}")


if __name__ == "__main__":
    main()
