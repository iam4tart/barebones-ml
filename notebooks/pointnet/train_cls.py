import os
import sys
import argparse
import torch
import torch.optim as optim
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(__file__))

from pointnet_cls     import PointNetCls
from pointnet_dataset import ModelNet40

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_points", type=int, default=1024)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--data_root", type=str, default="./data")
    parser.add_argument("--ckpt_dir", type=str, default="./checkpoints")
    parser.add_argument("--num_workers", type=int, default=4)
    return parser.parse_args()

def train_one_epoch(model, loader, optimizer, device):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    
    for pts, label in loader:
        pts = pts.to(device) # [B, N, 3]
        labels = labels.to(device) # [B]
        
        optimizer.zero_grad()
        
        logits, feat_trans = model(pts)
        loss, ce, reg = model.get_loss(logits, labels, feat_trans)
        
        loss.backward()
        optimizer.step()
        
        # total loss is loss average per sample in batch * batch size
        total_loss += loss.item() * pts.size(0)
        # logits.shape [B, num_classes]
        preds = logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += pts.size(0)
        
    return total_loss/total, correct/total

@torch.no_grad()
def eval_one_epoch(model, loader, device):
    model.eval()
    correct = 0
    total = 0
    
    for pts, labels in loader:
        pts = pts.to(device)
        labels = labels.to(device)
        
        logits, feat_trans = model(pts)
        preds = logits.argmax(dim=1)
        correct = (preds == labels).sum().item()
        total += pts.size(0)
        
    return correct/total

def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")
    
    train_ds = ModelNet40(args.data_root, "train", args.num_points, augment=True)
    test_ds = ModelNet40(args.data_root, "test", args.num_points, augment=False)
    
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    model = PointNetCls(num_classes=40, feature_transform=True).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"model params: {total_params:,}")
    
    