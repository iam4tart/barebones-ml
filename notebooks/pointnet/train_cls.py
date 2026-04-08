import os
import sys
import argparse
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(__file__))

from pointnet_cls     import PointNetCls
from pointnet_dataset import ModelNet40

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_points", type=int, default=1024)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--data_root", type=str, default="./datasets")
    parser.add_argument("--ckpt_dir", type=str, default="./notebooks/checkpoints")
    parser.add_argument("--num_workers", type=int, default=4)
    return parser.parse_args()

def train_one_epoch(model, loader, optimizer, device):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    
    pbar = tqdm(loader, desc="training", leave=False)
    for pts, label in loader:
        pts = pts.to(device) # [B, N, 3]
        labels = label.to(device) # [B]
        
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
        
        pbar.set_postfix({
            "loss": f"{loss.item():.4f}",
            "acc":  f"{correct/total*100:.1f}%"
        })
        
    return total_loss/total, correct/total

@torch.no_grad()
def eval_one_epoch(model, loader, device):
    model.eval()
    correct = 0
    total = 0
    
    pbar = tqdm(loader, desc="eval ", leave=False)
    for pts, labels in loader:
        pts = pts.to(device)
        labels = labels.to(device)
        
        logits, feat_trans = model(pts)
        preds = logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += pts.size(0)
        
        pbar.set_postfix({"acc": f"{correct/total*100:.1f}%"})
        
    return correct/total



def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")
    
    # release any cached gpu memory from previous runs
    if device.type == "cuda":
        torch.cuda.empty_cache()
        free, total = torch.cuda.mem_get_info()
        print(f"gpu memory free: {free/1024**2:.0f} MB / {total/1024**2:.0f} MB")
        
    train_ds = ModelNet40(args.data_root, "train", args.num_points, augment=True)
    test_ds = ModelNet40(args.data_root, "test", args.num_points, augment=False)
    
    pin_memory = device.type == "cuda"
    
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=pin_memory # parallel processing
    )
    
    test_loader = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=pin_memory
    )
    
    model = PointNetCls(num_classes=40, feature_transform=True).to(device)
    total_trainable_params = sum(p.numel() for p in model.parameters())
    print(f"model params: {total_trainable_params:,}")
    
    # steplr: 1/2 lr every 20 epochs - matches original pointnet paper
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)
    
    # diagnosis
    # print("\n--- test loader sanity check ---")
    # pts, labels = next(iter(test_loader))
    # print(f"pts shape:    {pts.shape}")
    # print(f"labels shape: {labels.shape}")
    # print(f"labels dtype: {labels.dtype}")
    # print(f"labels sample: {labels[:8]}")
    # print(f"labels min/max: {labels.min()} / {labels.max()}")
    # model.eval()
    # with torch.no_grad():
    #     pts = pts.to(device)
    #     labels = labels.to(device)
    #     logits, _ = model(pts)
    #     preds = logits.argmax(dim=1)
    #     print(f"preds sample:  {preds[:8]}")
    #     print(f"labels sample: {labels[:8]}")
    #     print(f"match: {(preds == labels).sum().item()} / {len(labels)}")
    # print("--- end sanity check ---\n")
    
    os.makedirs(args.ckpt_dir, exist_ok=True)
    best_acc = 0.0
    
    log_path = os.path.join(args.ckpt_dir, "training_log.csv")
    with open(log_path, "w") as f:
        f.write("epoch,train_loss,train_acc,test_acc,lr\n")
    
    for epoch in range(1, args.epochs+1):
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, device)
        test_acc = eval_one_epoch(model, test_loader, device)
        scheduler.step()
        
        current_lr = scheduler.get_last_lr()[0]
        
        with open(log_path, "a") as f:
            f.write(f"{epoch},{train_loss:.4f},{train_acc:.4f},{test_acc:.4f},{current_lr:.6f}\n")
        
        print(
            f"epoch {epoch:3d}/{args.epochs} | "
            f"loss {train_loss:.4f} | "
            f"train acc {train_acc*100:.1f}% | "
            f"test acc {test_acc*100:.1f}% | "
            f"lr {current_lr:.6f}"
        )
        
        # save best checkpoint
        if test_acc > best_acc:
            best_acc = test_acc
            ckpt_path = os.path.join(args.ckpt_dir, "pointnet_cls_best.pth")
            torch.save({
                "epoch": epoch,
                "state_dict": model.state_dict(),
                "test_acc": test_acc,
                "args": vars(args),
            }, ckpt_path)
            print(f" saved best checkpoint -> {ckpt_path}\n")
            
    print(f"best test accuracy: {best_acc*100:.1f}%")
    print(f"training log saved -> {log_path}")
            
if __name__ == "__main__":
    main()