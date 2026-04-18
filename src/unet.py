import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    """
    Two consecutive Conv2d + BatchNorm + ReLU blocks.
    This is the basic building block of U-Net.
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.block(x)


class Encoder(nn.Module):
    """
    Downsampling path — extracts features at increasing depth.
    Each step: DoubleConv → MaxPool (halves spatial dimensions)
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = DoubleConv(in_channels, out_channels)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        skip = self.conv(x)   # saved for skip connection
        down = self.pool(skip)
        return skip, down


class Decoder(nn.Module):
    """
    Upsampling path — reconstructs spatial resolution.
    Each step: ConvTranspose2d (doubles size) → concat skip → DoubleConv
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up   = nn.ConvTranspose2d(in_channels, in_channels // 2,
                                        kernel_size=2, stride=2)
        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x, skip):
        x = self.up(x)

        # Handle size mismatch due to odd dimensions
        if x.shape != skip.shape:
            x = F.interpolate(x, size=skip.shape[2:])

        x = torch.cat([skip, x], dim=1)  # concatenate along channel dim
        return self.conv(x)


class UNet(nn.Module):
    """
    U-Net for semantic segmentation.

    Architecture:
        Input → Enc1 → Enc2 → Enc3 → Enc4 → Bottleneck
                                              ↓
        Output ← Dec1 ← Dec2 ← Dec3 ← Dec4 ←┘
        (with skip connections from each encoder to matching decoder)

    Args:
        in_channels:  number of input channels (3 for RGB)
        num_classes:  number of output segmentation classes
        features:     channel sizes at each level
    """
    def __init__(self, in_channels=3, num_classes=21,
                 features=[64, 128, 256, 512]):
        super().__init__()

        self.encoders = nn.ModuleList()
        self.decoders = nn.ModuleList()

        # Encoder path
        ch = in_channels
        for f in features:
            self.encoders.append(Encoder(ch, f))
            ch = f

        # Bottleneck — deepest layer, no pooling
        self.bottleneck = DoubleConv(features[-1], features[-1] * 2)

        # Decoder path (reversed)
        for f in reversed(features):
            self.decoders.append(Decoder(f * 2, f))

        # Final 1x1 conv — maps to num_classes
        self.final_conv = nn.Conv2d(features[0], num_classes, kernel_size=1)

    def forward(self, x):
        skips = []

        # Encoder
        for enc in self.encoders:
            skip, x = enc(x)
            skips.append(skip)

        # Bottleneck
        x = self.bottleneck(x)

        # Decoder — use skip connections in reverse order
        for dec, skip in zip(self.decoders, reversed(skips)):
            x = dec(x, skip)

        return self.final_conv(x)


if __name__ == '__main__':
    # Quick sanity check
    model  = UNet(in_channels=3, num_classes=21)
    x      = torch.randn(2, 3, 256, 256)  # batch of 2 RGB images
    output = model(x)
    print(f"Input:  {x.shape}")
    print(f"Output: {output.shape}")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")