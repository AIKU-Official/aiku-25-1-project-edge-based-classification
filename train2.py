import argparse
import time
import copy

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
from torch.utils.data import DataLoader
from tqdm import tqdm


class SobelEdge(object):
    """Compute a single‑channel Sobel edge map and concat with the image."""

    def __call__(self, img):
        # PIL -> numpy (H,W,3) in [0,255]
        x = np.array(img)
        gray = cv2.cvtColor(x, cv2.COLOR_RGB2GRAY)
        sobelx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        sobely = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        edge = cv2.magnitude(sobelx, sobely)
        edge = (edge - edge.min()) / (edge.ptp() + 1e-6)  # 0‑1
        edge = edge[..., None]  # (H,W,1)
        x = x.astype(np.float32) / 255.0
        x = np.concatenate([x, edge], axis=2)  # (H,W,4)
        return torch.from_numpy(x.transpose(2, 0, 1))  # C,H,W


IMAGENET_MEAN = [0.485, 0.456, 0.406, 0.5]
IMAGENET_STD = [0.229, 0.224, 0.225, 0.25]


def parse_args():
    p = argparse.ArgumentParser("Fine‑tune ResNet‑34 + edges")
    p.add_argument("--data-dir", required=True)
    p.add_argument("--log-dir", required=True)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--momentum", type=float, default=0.9)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--pretrained", action="store_true")
    p.add_argument("--full-finetuning", action="store_true")
    p.add_argument("--model-path")
    p.add_argument("--device", default="cuda")
    p.add_argument("--save-path", default="resnet34_edges.pth")
    return p.parse_args()


def build_dataloaders(root, bs, workers=4):
    train_aug = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        SobelEdge(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    val_aug = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        SobelEdge(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])

    train_ds = datasets.ImageFolder(f"{root}/train", train_aug)
    val_ds = datasets.ImageFolder(f"{root}/val", val_aug)

    return (
        DataLoader(train_ds, bs, True, num_workers=workers, pin_memory=True),
        DataLoader(val_ds, bs, False, num_workers=workers, pin_memory=True),
        len(train_ds.classes),
    )


def modify_first_conv(m):
    # Expand conv1 weight to 4 input channels
    old_w = m.conv1.weight
    new_w = torch.zeros(old_w.size(0), 4, *old_w.shape[2:])
    new_w[:, :3] = old_w
    new_w[:, 3] = old_w.mean(1)
    m.conv1 = nn.Conv2d(4, 64, kernel_size=7, stride=2, padding=3, bias=False)
    m.conv1.weight = nn.Parameter(new_w)
    return m


def train_one_epoch(model, loader, crit, opt, device):
    model.train()
    tloss = tcorrect = total = 0
    for x, y in tqdm(loader, leave=False):
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        opt.zero_grad()
        out = model(x)
        loss = crit(out, y)
        loss.backward()
        opt.step()
        tloss += loss.item() * x.size(0)
        tcorrect += (out.argmax(1) == y).sum().item()
        total += x.size(0)
    return tloss / total, tcorrect / total


def validate(model, loader, crit, device):
    model.eval()
    vloss = vcorrect = total = 0
    with torch.no_grad():
        for x, y in tqdm(loader, leave=False):
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            out = model(x)
            loss = crit(out, y)
            vloss += loss.item() * x.size(0)
            vcorrect += (out.argmax(1) == y).sum().item()
            total += x.size(0)
    return vloss / total, vcorrect / total


def main():
    args = parse_args()
    dev = torch.device(args.device if torch.cuda.is_available() else "cpu")

    tr_loader, val_loader, n_cls = build_dataloaders(args.data_dir, args.batch_size)

    model = models.resnet34(pretrained=args.pretrained)
    model = modify_first_conv(model)

    if args.model_path:
        ckpt = torch.load(args.model_path, map_location="cpu")
        model.load_state_dict(ckpt.get("state_dict", ckpt))

    model.fc = nn.Linear(model.fc.in_features, n_cls)

    # param selection
    if args.full_finetuning:
        params = model.parameters()
    else:
        for p in model.parameters():
            p.requires_grad = False
        for p in model.fc.parameters():
            p.requires_grad = True
        for p in model.conv1.parameters():
            p.requires_grad = True  # need to update first conv because of new channel
        params = filter(lambda p: p.requires_grad, model.parameters())

    model.to(dev)

    criterion = nn.CrossEntropyLoss()
    opt = optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay)
    sched = optim.lr_scheduler.StepLR(opt, 7, 0.1)

    best_acc = 0
    best_wts = copy.deepcopy(model.state_dict())

    for epoch in range(1, args.epochs + 1):
        print(f"\nEpoch {epoch}/{args.epochs}")
        t_loss, t_acc = train_one_epoch(model, tr_loader, criterion, opt, dev)
        v_loss, v_acc = validate(model, val_loader, criterion, dev)
        sched.step()
        print(f"  Train loss {t_loss:.4f} acc {t_acc:.4f}")
        print(f"  Val   loss {v_loss:.4f} acc {v_acc:.4f}")

        if v_acc > best_acc:
            best_acc = v_acc
            best_wts = copy.deepcopy(model.state_dict())

    model.load_state_dict(best_wts)
    torch.save(model.state_dict(), args.save_path)
    print(f"Best val acc: {best_acc:.4f}. Saved to {args.save_path}")


if __name__ == "__main__":
    main()
