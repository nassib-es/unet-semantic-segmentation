import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
import numpy as np
import os
import sys
sys.path.append('.')

from src.unet import UNet
from src.dataset import VOCDataset


def iou_score(preds, masks, num_classes=21, ignore_index=255):
    """
    Intersection over Union — the standard metric for segmentation.
    Better than accuracy because it handles class imbalance.
    """
    ious = []
    preds = preds.view(-1)
    masks = masks.view(-1)

    for cls in range(num_classes):
        pred_cls = (preds == cls)
        true_cls = (masks == cls)
        intersection = (pred_cls & true_cls).sum().float()
        union        = (pred_cls | true_cls).sum().float()
        if union == 0:
            continue  # class not present in batch, skip
        ious.append((intersection / union).item())

    return np.mean(ious) if ious else 0.0


def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    total_iou  = 0

    for imgs, masks in tqdm(loader, desc='Training', leave=False):
        imgs  = imgs.to(device)
        masks = masks.to(device)

        optimizer.zero_grad()
        outputs = model(imgs)
        loss    = criterion(outputs, masks)
        loss.backward()
        optimizer.step()

        preds = outputs.argmax(dim=1)
        total_loss += loss.item()
        total_iou  += iou_score(preds.cpu(), masks.cpu())

    n = len(loader)
    return total_loss / n, total_iou / n


def validate(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    total_iou  = 0

    with torch.no_grad():
        for imgs, masks in tqdm(loader, desc='Validating', leave=False):
            imgs  = imgs.to(device)
            masks = masks.to(device)

            outputs = model(imgs)
            loss    = criterion(outputs, masks)
            preds   = outputs.argmax(dim=1)

            total_loss += loss.item()
            total_iou  += iou_score(preds.cpu(), masks.cpu())

    n = len(loader)
    return total_loss / n, total_iou / n


def train(epochs=30, batch_size=8, lr=1e-4):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Dataset
    full_dataset = VOCDataset(root='data', split='train', augment=True)
    val_size     = int(0.15 * len(full_dataset))
    train_size   = len(full_dataset) - val_size
    train_ds, val_ds = random_split(full_dataset, [train_size, val_size])

    train_loader = DataLoader(train_ds, batch_size=batch_size,
                              shuffle=True,  num_workers=0, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size,
                              shuffle=False, num_workers=0, pin_memory=True)

    print(f"Train: {train_size} | Val: {val_size}")

    # Model
    import segmentation_models_pytorch as smp
    model = smp.Unet(
        encoder_name    = "resnet50",
        encoder_weights = "imagenet",
        in_channels     = 3,
        classes         = 21
    ).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=3, factor=0.5)

    best_iou = 0.0
    history  = {'train_loss': [], 'val_loss': [],
                 'train_iou': [],  'val_iou': []}

    print("=" * 60)
    print(f"Training U-Net | Epochs: {epochs} | Batch: {batch_size} | LR: {lr}")
    print("=" * 60)

    for epoch in range(1, epochs + 1):
        train_loss, train_iou = train_one_epoch(
            model, train_loader, optimizer, criterion, device)
        val_loss, val_iou = validate(
            model, val_loader, criterion, device)

        scheduler.step(val_loss)

        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_iou'].append(train_iou)
        history['val_iou'].append(val_iou)

        print(f"Ep {epoch:3d}/{epochs} | "
              f"Train Loss: {train_loss:.4f} IoU: {train_iou:.4f} | "
              f"Val Loss: {val_loss:.4f} IoU: {val_iou:.4f}")

        # Save best model
        if val_iou > best_iou:
            best_iou = val_iou
            os.makedirs('models', exist_ok=True)
            torch.save(model.state_dict(), 'models/best_model_resnet50.pth')
            print(f"  ✓ Saved best model (IoU: {best_iou:.4f})")

    # Save history
    np.save('models/history.npy', history, allow_pickle=True)
    print(f"\nTraining complete! Best Val IoU: {best_iou:.4f}")
    return model, history


if __name__ == '__main__':
    train(epochs=60, batch_size=8, lr=1e-4)