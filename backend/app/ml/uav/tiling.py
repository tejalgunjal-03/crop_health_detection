"""Tile a large RGB image into 224x224 patches, predict each, and stitch the result.

The farmer never splits images manually. Overlapping tiles are blended with a
triangular weight window so tile borders do not show seams.
NumPy only (no torch) so it can be unit-tested with any predictor function.
"""
from __future__ import annotations
from typing import Callable
import numpy as np

PredictFn = Callable[[np.ndarray], np.ndarray]    # (N,T,T,3) uint8 -> (N,C,T,T) float32 probabilities


def _starts(length: int, tile: int, stride: int) -> list[int]:
    if length <= tile:
        return [0]
    s = list(range(0, length - tile + 1, stride))
    if s[-1] != length - tile:
        s.append(length - tile)        # last tile aligned to the border, no padding needed
    return s


def _window(tile: int) -> np.ndarray:
    ramp = 1 - np.abs(np.linspace(-1, 1, tile))
    w = np.outer(ramp, ramp).astype(np.float32)
    return np.clip(w, 0.05, None)      # never zero, so every pixel gets weight


def segment_large_image(rgb: np.ndarray, predict_fn: PredictFn, num_classes: int,
                        tile: int = 224, overlap: float = 0.5, batch_size: int = 4):
    """Return (label_map HxW uint8, confidence HxW float32, n_tiles)."""
    h, w, _ = rgb.shape
    pad_h, pad_w = max(0, tile - h), max(0, tile - w)
    if pad_h or pad_w:                                # only when image is smaller than one tile
        rgb = np.pad(rgb, ((0, pad_h), (0, pad_w), (0, 0)), mode="reflect")
    H, W, _ = rgb.shape
    stride = max(1, int(tile * (1 - overlap)))
    ys, xs = _starts(H, tile, stride), _starts(W, tile, stride)
    coords = [(y, x) for y in ys for x in xs]
    acc = np.zeros((num_classes, H, W), np.float32)
    wsum = np.zeros((H, W), np.float32)
    win = _window(tile)
    for i in range(0, len(coords), batch_size):
        chunk = coords[i:i + batch_size]
        batch = np.stack([rgb[y:y + tile, x:x + tile] for y, x in chunk])
        probs = predict_fn(batch)
        for (y, x), p in zip(chunk, probs):
            acc[:, y:y + tile, x:x + tile] += p * win
            wsum[y:y + tile, x:x + tile] += win
    acc /= wsum[None]
    labels = acc.argmax(0).astype(np.uint8)[:h, :w]
    conf = acc.max(0)[:h, :w]
    return labels, conf, len(coords)
