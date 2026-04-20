import torch
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.datasets import VOCSegmentation
import numpy as np
from PIL import Image


# PASCAL VOC 2012 — 21 classes including background
VOC_CLASSES = [
    'background', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle',
    'bus', 'car', 'cat', 'chair', 'cow', 'diningtable', 'dog',
    'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa',
    'train', 'tvmonitor'
]

# Color map for visualization — each class gets a distinct color
VOC_COLORMAP = [
    [0, 0, 0],        # background — black
    [128, 0, 0],      # aeroplane — dark red
    [0, 128, 0],      # bicycle — dark green
    [128, 128, 0],    # bird — olive
    [0, 0, 128],      # boat — dark blue
    [128, 0, 128],    # bottle — purple
    [0, 128, 128],    # bus — teal
    [128, 128, 128],  # car — gray
    [64, 0, 0],       # cat — dark brown
    [192, 0, 0],      # chair — red
    [64, 128, 0],     # cow — green
    [192, 128, 0],    # diningtable — orange
    [64, 0, 128],     # dog — dark purple
    [192, 0, 128],    # horse — pink
    [64, 128, 128],   # motorbike — cyan
    [192, 128, 128],  # person — light pink
    [0, 64, 0],       # pottedplant — forest green
    [128, 64, 0],     # sheep — brown
    [0, 192, 0],      # sofa — bright green
    [128, 192, 0],    # train — yellow-green
    [0, 64, 128],     # tvmonitor — steel blue
]


def mask_to_tensor(mask):
    """
    Convert PIL segmentation mask to tensor of class indices.
    Pixels with value 255 (boundary/ignore) are mapped to 0 (background).
    """
    mask = np.array(mask)
    mask[mask == 255] = 0
    return torch.from_numpy(mask).long()


class VOCDataset(Dataset):
    """
    PASCAL VOC 2012 Segmentation Dataset wrapper.
    Downloads automatically on first use.
    """

    def __init__(self, root='data', split='train',
             image_size=(256, 256), augment=False):
        self.image_size = image_size
        self.augment    = augment

        self.dataset = VOCSegmentation(
            root=root,
            year='2012',
            image_set=split,
            download=True
        )

        # Base transforms
        self.img_transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std= [0.229, 0.224, 0.225]
            )
        ])

        self.mask_transform = transforms.Compose([
            transforms.Resize(image_size,
                            interpolation=transforms.InterpolationMode.NEAREST),
        ])

        # Augmentation transforms — applied to both image and mask together
        self.aug_transform = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.1),
            transforms.RandomRotation(degrees=15),
        ])

        # Color jitter — image only
        self.color_jitter = transforms.ColorJitter(
            brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1
        )

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img, mask = self.dataset[idx]

        if self.augment:
            # Apply same spatial transform to both image and mask
            seed = torch.randint(0, 2**32, (1,)).item()

            torch.manual_seed(seed)
            img  = self.aug_transform(img)
            torch.manual_seed(seed)
            mask = self.aug_transform(mask)

            # Color jitter on image only
            img = self.color_jitter(img)

        img  = self.img_transform(img)
        mask = self.mask_transform(mask)
        mask = mask_to_tensor(mask)

        return img, mask

    @staticmethod
    def decode_mask(mask_tensor):
        """
        Convert class index tensor to RGB image for visualization.
        mask_tensor: H x W tensor of class indices
        """
        mask = mask_tensor.numpy()
        rgb  = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
        for cls_idx, color in enumerate(VOC_COLORMAP):
            rgb[mask == cls_idx] = color
        return rgb


if __name__ == '__main__':
    print("Loading PASCAL VOC 2012...")
    dataset = VOCDataset(root='data', split='train')
    print(f"Training samples: {len(dataset)}")

    img, mask = dataset[0]
    print(f"Image shape: {img.shape}")
    print(f"Mask shape:  {mask.shape}")
    print(f"Unique classes in sample: {mask.unique().tolist()}")