"""Leaf disease inference using the saved checkpoint."""
from __future__ import annotations
import torch
from PIL import Image
from ..core.config import get_settings
from ..ml.leaf.model import eval_transform
from .model_loader import load_checkpoint


def pretty(name: str) -> str:
    return name.replace("_", " ")


def predict_leaf(img: Image.Image) -> dict:
    from ..core.errors import AppError
    model, ckpt = load_checkpoint(get_settings().leaf_model_path, "leaf")
    tf = eval_transform(ckpt["img_size"], ckpt["mean"], ckpt["std"])
    with torch.no_grad():
        probs = torch.softmax(model(tf(img.convert("RGB")).unsqueeze(0)), dim=1)[0].tolist()
    names = [pretty(n) for n in ckpt["class_names"]]
    best = max(range(len(probs)), key=probs.__getitem__)
    confidence = float(probs[best])
    
    # If the model is not highly confident, reject it (helps filter out other crop leaves)
    if confidence < 0.65:
        raise AppError("this model is not trained for this leaf data", "invalid_leaf", 400)
        
    return {"disease": names[best], "confidence": confidence,
            "probabilities": {n: float(p) for n, p in zip(names, probs)},
            "model_arch": ckpt.get("arch", "mobilenet_v3_small")}
