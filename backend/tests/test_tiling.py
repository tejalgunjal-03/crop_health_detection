import numpy as np
from app.ml.uav.tiling import segment_large_image
from app.ml.uav.regions import find_regions, make_overlay

CLASSES = {0: {"name": "bg", "label": "bg", "color": [0, 0, 0]}, 1: {"name": "bad", "label": "Bad", "color": [255, 0, 0]}}

def fake_predict(batch):
    """Per-pixel rule (so stitching must reproduce it exactly): red channel > 128 => class 1."""
    p1 = (batch[..., 0] > 128).astype(np.float32)
    return np.stack([1 - p1, p1], axis=1)

def _field(h, w):
    rgb = np.zeros((h, w, 3), np.uint8); rgb[100:300, 150:420, 0] = 255; return rgb

def test_stitching_matches_per_pixel_truth_for_odd_sizes():
    for h, w in [(500, 733), (224, 224), (225, 449), (1000, 640)]:
        rgb = _field(h, w)
        labels, conf, n = segment_large_image(rgb, fake_predict, 2, batch_size=3)
        assert labels.shape == (h, w)
        assert np.array_equal(labels, (rgb[..., 0] > 128).astype(np.uint8)), (h, w)
        assert n >= 1

def test_image_smaller_than_tile():
    rgb = _field(300, 300)[:150, :180]
    labels, _, n = segment_large_image(rgb, fake_predict, 2)
    assert labels.shape == (150, 180) and n == 1

def test_regions_and_overlay():
    rgb = _field(500, 733); labels = (rgb[..., 0] > 128).astype(np.uint8)
    regs = find_regions(labels, [1], CLASSES, min_area_px=400)
    assert len(regs) == 1 and regs[0]["width"] >= 260
    ov = make_overlay(rgb, labels, CLASSES, [1], regs)
    assert ov.shape == rgb.shape
    assert find_regions(np.zeros((100, 100), np.uint8), [1], CLASSES, 10) == []
