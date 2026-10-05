"""UAV RGB field-image segmentation: tile -> U-Net -> stitch -> suspicious regions.
Output is LOCALISATION ONLY. It is never passed to data fusion."""
from __future__ import annotations
import uuid
import cv2
import numpy as np
import torch
from PIL import Image
from ..core.config import get_settings, load_yaml
from ..ml.uav.regions import class_fractions, find_regions, make_overlay
from ..ml.uav.tiling import segment_large_image
from .model_loader import load_checkpoint


def load_classes() -> tuple[dict, list[int]]:
    cfg = load_yaml(get_settings().uav_classes_path)
    return {int(k): v for k, v in cfg["classes"].items()}, list(cfg["suspicious_class_ids"])


def analyze_field(img: Image.Image) -> dict:
    s = get_settings()
    model, ckpt = load_checkpoint(s.uav_model_path, "uav")
    classes, suspicious = load_classes()
    tile = ckpt.get("tile_size", 224)
    
    # Scale down the image to have a max dimension of ~1024 to match the dataset's GSD (Ground Sample Distance)
    # The U-Net was trained on 224x224 patches. If a 4K image is uploaded, a 224x224 crop contains almost no context.
    MAX_DIM = 1200
    if max(img.size) > MAX_DIM:
        img.thumbnail((MAX_DIM, MAX_DIM), Image.Resampling.LANCZOS)
        
    rgb = np.asarray(img.convert("RGB"))

    def predict(batch: np.ndarray) -> np.ndarray:
        x = torch.from_numpy(batch).permute(0, 3, 1, 2).float() / 255.0
        with torch.no_grad():
            return torch.softmax(model(x), dim=1).numpy()

    labels, conf, n_tiles = segment_large_image(rgb, predict, ckpt["num_classes"], tile=tile, overlap=0.5, batch_size=4)
    
    # Removed medianBlur as it corrupts integer class indices and erases minority classes.
    
    regions = find_regions(labels, suspicious, classes, s.uav_min_region_px)
    overlay = make_overlay(rgb, labels, classes, suspicious, regions)
    out_dir = s.results_dir / "uav"; out_dir.mkdir(parents=True, exist_ok=True)
    name = f"{uuid.uuid4().hex}_overlay.png"
    cv2.imwrite(str(out_dir / name), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
    fr = class_fractions(labels, classes)
    return {
        "overlay_file": name, "image_width": rgb.shape[1], "image_height": rgb.shape[0], "n_tiles": n_tiles,
        "class_fractions": fr,
        "suspicious_pixel_fraction": round(sum(fr[classes[k]["name"]] for k in suspicious), 5),
        "regions": regions, "mean_pixel_confidence": round(float(conf.mean()), 4),
        "legend": {classes[k]["label"]: classes[k]["color"] for k in classes},
        "note": ("Model-estimated from RGB tiles. This highlights where to inspect; it is NOT a crop-health diagnosis. "
                 "Accuracy on field images is unverified: the model was trained on 224x224 dataset patches."),
    }
