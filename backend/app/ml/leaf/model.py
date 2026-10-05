"""Leaf classifier: MobileNetV3-Small transfer learning (small, CPU friendly)."""
from __future__ import annotations
import torch.nn as nn
from torchvision import models, transforms

IMAGENET_MEAN, IMAGENET_STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]


def build_leaf_model(num_classes: int, pretrained: bool = True) -> nn.Module:
    weights = models.MobileNet_V3_Small_Weights.IMAGENET1K_V1 if pretrained else None
    m = models.mobilenet_v3_small(weights=weights)
    m.classifier[-1] = nn.Linear(m.classifier[-1].in_features, num_classes)
    return m


def eval_transform(size: int = 224, mean=IMAGENET_MEAN, std=IMAGENET_STD):
    return transforms.Compose([transforms.Resize((size, size)), transforms.ToTensor(),
                               transforms.Normalize(mean, std)])


def train_transform(size: int = 224, mean=IMAGENET_MEAN, std=IMAGENET_STD):
    """On-the-fly augmentation (training split ONLY)."""
    return transforms.Compose([
        transforms.RandomResizedCrop(size, scale=(0.7, 1.0)),
        transforms.RandomHorizontalFlip(), transforms.RandomVerticalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.15),
        transforms.ToTensor(), transforms.Normalize(mean, std)])
