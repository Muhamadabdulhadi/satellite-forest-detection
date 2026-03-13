import torch
import torch.nn as nn


class DoubleConv(nn.Module):

    def __init__(self, in_c, out_c):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_c, out_c, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(out_c, out_c, 3, padding=1),
            nn.ReLU()
        )

    def forward(self, x):
        return self.conv(x)


class UNet(nn.Module):

    def __init__(self):
        super().__init__()

        self.down1 = DoubleConv(3, 64)
        self.pool = nn.MaxPool2d(2)

        self.down2 = DoubleConv(64, 128)

        self.up = nn.ConvTranspose2d(128, 64, 2, stride=2)

        self.out = nn.Conv2d(64, 1, 1)

    def forward(self, x):

        x1 = self.down1(x)
        x2 = self.pool(x1)

        x3 = self.down2(x2)

        x4 = self.up(x3)

        x5 = x4 + x1

        return torch.sigmoid(self.out(x5))