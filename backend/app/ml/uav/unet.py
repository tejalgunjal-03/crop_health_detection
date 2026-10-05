"""Lightweight U-Net for 224x224 RGB -> per-pixel class logits. Trained from scratch (CPU)."""
from __future__ import annotations
import torch
import torch.nn as nn


def _block(i: int, o: int) -> nn.Sequential:
    return nn.Sequential(nn.Conv2d(i, o, 3, padding=1, bias=False), nn.BatchNorm2d(o), nn.ReLU(inplace=True),
                         nn.Conv2d(o, o, 3, padding=1, bias=False), nn.BatchNorm2d(o), nn.ReLU(inplace=True))


class UNet(nn.Module):
    def __init__(self, num_classes: int, base: int = 16, in_channels: int = 3):
        super().__init__()
        c = [base, base * 2, base * 4, base * 8, base * 16]
        self.enc = nn.ModuleList([_block(in_channels, c[0]), _block(c[0], c[1]), _block(c[1], c[2]), _block(c[2], c[3])])
        self.pool = nn.MaxPool2d(2)
        self.bott = _block(c[3], c[4])
        self.up = nn.ModuleList([nn.ConvTranspose2d(c[4], c[3], 2, 2), nn.ConvTranspose2d(c[3], c[2], 2, 2),
                                 nn.ConvTranspose2d(c[2], c[1], 2, 2), nn.ConvTranspose2d(c[1], c[0], 2, 2)])
        self.dec = nn.ModuleList([_block(c[3] * 2, c[3]), _block(c[2] * 2, c[2]), _block(c[1] * 2, c[1]), _block(c[0] * 2, c[0])])
        self.head = nn.Conv2d(c[0], num_classes, 1)

    def forward(self, x):
        skips = []
        for e in self.enc:
            x = e(x); skips.append(x); x = self.pool(x)
        x = self.bott(x)
        for up, dec, s in zip(self.up, self.dec, reversed(skips)):
            x = dec(torch.cat([up(x), s], dim=1))
        return self.head(x)
