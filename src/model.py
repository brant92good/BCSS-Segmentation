"""U-Net model architecture with attention mechanism."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    """Block with two convolutional blocks."""

    def __init__(self, in_channels, out_channels, mid_channels=None, dropout_rate=0.1):
        """
        Double convolution.

        Args:
            in_channels: Number of input channels
            out_channels: Number of output channels
            mid_channels: Number of intermediate channels
            dropout_rate: Dropout probability
        """
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels

        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
            nn.Dropout2d(p=dropout_rate),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(negative_slope=0.01, inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downsampling block using stride convolution."""

    def __init__(self, in_channels, out_channels):
        """
        Down block.

        Args:
            in_channels: Number of input channels
            out_channels: Number of output channels
        """
        super().__init__()
        self.downsample = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=2, stride=2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.downsample(x)


class AttentionBlock(nn.Module):
    """Attention mechanism for skip connections."""

    def __init__(self, F_g, F_l, F_int):
        """
        Attention block.
        
        Args:
            F_g: Number of gate (upsampled) feature channels
            F_l: Number of skip connection feature channels
            F_int: Number of intermediate attention channels
        """
        super().__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )
        
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )
        
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        return x * psi


class Up(nn.Module):
    """Upsampling block with attention."""

    def __init__(self, in_channels, out_channels, skip_channels=None, use_attention=True):
        """
        Up block with attention.

        Args:
            in_channels: Number of input channels
            out_channels: Number of output channels
            skip_channels: Number of skip connection channels
            use_attention: Whether to use attention mechanism
        """
        super().__init__()

        if skip_channels is None:
            skip_channels = in_channels
        
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(in_channels, in_channels // 2, kernel_size=3, padding=1)
        )
        
        self.use_attention = use_attention
        if use_attention:
            self.attention = AttentionBlock(
                F_g=in_channels // 2,
                F_l=skip_channels,
                F_int=in_channels // 4
            )
        
        concat_channels = (in_channels // 2) + skip_channels
        self.conv = DoubleConv(concat_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
        
        if self.use_attention:
            x2 = self.attention(g=x1, x=x2)
        
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    """Final output convolution."""

    def __init__(self, in_channels, out_channels):
        """
        Output convolution.

        Args:
            in_channels: Number of input channels
            out_channels: Number of output channels
        """
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class UNet(nn.Module):
    """U-Net model with attention and improved architecture."""

    def __init__(self, n_channels, n_classes, use_attention=True):
        """
        Initialize U-Net model.
        
        Args:
            n_channels: Number of input channels
            n_classes: Number of output classes
            use_attention: Whether to use attention mechanism
        """
        super().__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes

        # Encoder
        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        self.down4 = Down(512, 512)
        
        # Decoder
        self.up1 = Up(in_channels=512, out_channels=256, skip_channels=512, use_attention=use_attention)
        self.up2 = Up(in_channels=256, out_channels=128, skip_channels=256, use_attention=use_attention)
        self.up3 = Up(in_channels=128, out_channels=64, skip_channels=128, use_attention=use_attention)
        self.up4 = Up(in_channels=64, out_channels=64, skip_channels=64, use_attention=use_attention)
        
        self.outc = OutConv(64, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits
