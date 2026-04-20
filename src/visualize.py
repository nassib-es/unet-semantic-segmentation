import sys
sys.path.append('.')
import torch
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms
from PIL import Image
import os
import segmentation_models_pytorch as smp

from src.unet import UNet
from src.dataset import VOCDataset, VOC_CLASSES

def visualize_predictions(num_samples=6):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load model
    model = smp.Unet(
        encoder_name    = "resnet50",
        encoder_weights = None,
        in_channels     = 3,
        classes         = 21
    ).to(device)
    model.load_state_dict(torch.load('models/best_model_resnet50.pth',
                                  map_location=device))
    model.eval()

    # Load validation data
    dataset = VOCDataset(root='data', split='val')
    indices = np.random.choice(len(dataset), num_samples, replace=False)

    fig, axes = plt.subplots(num_samples, 3, figsize=(12, num_samples * 4))
    fig.patch.set_facecolor('#0A0E14')

    # Denormalize for display
    inv_normalize = transforms.Normalize(
        mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225],
        std=[1/0.229, 1/0.224, 1/0.225]
    )

    for row, idx in enumerate(indices):
        img, mask = dataset[idx]

        # Predict
        with torch.no_grad():
            output = model(img.unsqueeze(0).to(device))
            pred   = output.argmax(dim=1).squeeze(0).cpu()

        # Denormalize image for display
        img_display = inv_normalize(img).permute(1, 2, 0).clamp(0, 1).numpy()

        # Decode masks to RGB
        pred_rgb = VOCDataset.decode_mask(pred)
        true_rgb = VOCDataset.decode_mask(mask)

        # Plot
        for col, (data, title) in enumerate([
            (img_display, 'Input Image'),
            (true_rgb,    'Ground Truth'),
            (pred_rgb,    'Prediction')
        ]):
            ax = axes[row, col]
            ax.imshow(data)
            ax.set_title(title, color='#00D4FF', fontsize=10)
            ax.axis('off')

        # Show classes present
        classes_present = [VOC_CLASSES[c] for c in mask.unique().tolist() if c < 21]
        axes[row, 0].set_xlabel(
            f"Classes: {', '.join(classes_present)}",
            color='#7A9CC0', fontsize=8
        )

    plt.suptitle('U-Net Semantic Segmentation — PASCAL VOC 2012',
             color='#00D4FF', fontsize=13, fontweight='bold', y=1.02)
    plt.tight_layout()
    os.makedirs('docs', exist_ok=True)
    plt.savefig('docs/predictions.png', dpi=150,
                bbox_inches='tight', facecolor='#0A0E14')
    plt.show()
    print("Saved to docs/predictions.png")

if __name__ == '__main__':
    visualize_predictions()