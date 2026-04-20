import numpy as np
import matplotlib.pyplot as plt
import torch
from src.dataset import VOCDataset, VOC_CLASSES
import segmentation_models_pytorch as smp


def plot_training_history(history_path='models/history.npy',
                          save_path='docs/training_curves.png'):
    """Plot training and validation loss and IoU curves."""

    history = np.load(history_path, allow_pickle=True).item()

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.patch.set_facecolor('#0A0E14')

    for ax in axes:
        ax.set_facecolor('#0F1520')
        ax.tick_params(colors='#7A9CC0')
        for spine in ax.spines.values():
            spine.set_edgecolor('#1E2D45')

    epochs = range(1, len(history['train_loss']) + 1)

    # Loss
    axes[0].plot(epochs, history['train_loss'], color='#00D4FF',
                 label='Train Loss', linewidth=2)
    axes[0].plot(epochs, history['val_loss'], color='#F06292',
                 label='Val Loss', linewidth=2)
    axes[0].set_title('Loss', color='#00D4FF', fontsize=12)
    axes[0].set_xlabel('Epoch', color='#7A9CC0')
    axes[0].set_ylabel('Cross Entropy Loss', color='#7A9CC0')
    axes[0].legend(facecolor='#141C2B', labelcolor='#E8EEF7')

    # IoU
    axes[1].plot(epochs, history['train_iou'], color='#00D4FF',
                 label='Train IoU', linewidth=2)
    axes[1].plot(epochs, history['val_iou'], color='#81C784',
                 label='Val IoU', linewidth=2)
    axes[1].set_title('Mean IoU', color='#00D4FF', fontsize=12)
    axes[1].set_xlabel('Epoch', color='#7A9CC0')
    axes[1].set_ylabel('mIoU', color='#7A9CC0')
    axes[1].legend(facecolor='#141C2B', labelcolor='#E8EEF7')

    plt.suptitle('U-Net ResNet50 — Training History',
                 color='#00D4FF', fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150,
                bbox_inches='tight', facecolor='#0A0E14')
    plt.show()
    print(f"Saved to {save_path}")


def per_class_iou(model_path='models/best_model_resnet50.pth',
                  save_path='docs/per_class_iou.png'):
    """Compute and plot IoU per class."""

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = smp.Unet(
        encoder_name='resnet50',
        encoder_weights=None,
        in_channels=3,
        classes=21
    ).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    dataset = VOCDataset(root='data', split='val')
    class_intersection = np.zeros(21)
    class_union        = np.zeros(21)

    with torch.no_grad():
        for img, mask in dataset:
            output = model(img.unsqueeze(0).to(device))
            pred   = output.argmax(dim=1).squeeze(0).cpu().numpy()
            mask   = mask.numpy()

            for cls in range(21):
                pred_cls = (pred == cls)
                true_cls = (mask == cls)
                class_intersection[cls] += (pred_cls & true_cls).sum()
                class_union[cls]        += (pred_cls | true_cls).sum()

    iou_per_class = np.where(
        class_union > 0,
        class_intersection / class_union,
        np.nan
    )

    # Plot
    valid = ~np.isnan(iou_per_class)
    names = [VOC_CLASSES[i] for i in range(21) if valid[i]]
    ious  = iou_per_class[valid]
    colors = ['#81C784' if iou > 0.3 else '#FFB74D' if iou > 0.15 else '#F06292'
              for iou in ious]

    fig, ax = plt.subplots(figsize=(14, 6))
    fig.patch.set_facecolor('#0A0E14')
    ax.set_facecolor('#0F1520')
    ax.tick_params(colors='#7A9CC0')
    for spine in ax.spines.values():
        spine.set_edgecolor('#1E2D45')

    bars = ax.bar(names, ious, color=colors)
    ax.axhline(np.nanmean(iou_per_class), color='#00D4FF',
               linestyle='--', label=f'Mean IoU: {np.nanmean(iou_per_class):.3f}')
    ax.set_title('Per-class IoU — Val Set', color='#00D4FF', fontsize=12)
    ax.set_ylabel('IoU', color='#7A9CC0')
    plt.xticks(rotation=45, ha='right')
    ax.legend(facecolor='#141C2B', labelcolor='#E8EEF7')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150,
                bbox_inches='tight', facecolor='#0A0E14')
    plt.show()
    print(f"Saved to {save_path}")
    print("\nPer-class IoU:")
    for name, iou in zip(names, ious):
        bar = '█' * int(iou * 30)
        print(f"  {name:<15} {iou:.3f} {bar}")


if __name__ == '__main__':
    plot_training_history()
    per_class_iou()