"""Load trained checkpoints (never trains). Cached; reloaded if the file changes."""
from __future__ import annotations
from pathlib import Path
from ..core.errors import ModelNotFoundError, ModelLoadError

_cache: dict = {}


def load_checkpoint(path: Path, which: str):
    import torch
    path = Path(path)
    if not path.exists():
        raise ModelNotFoundError(which)
    key = (str(path), path.stat().st_mtime)
    if key in _cache:
        return _cache[key]
    try:
        ckpt = torch.load(path, map_location="cpu", weights_only=True)
        if which == "leaf":
            from ..ml.leaf.model import build_leaf_model
            model = build_leaf_model(len(ckpt["class_names"]), pretrained=False)
        else:
            from ..ml.uav.unet import UNet
            model = UNet(ckpt["num_classes"], base=ckpt.get("base_channels", 16))
        model.load_state_dict(ckpt["model_state"])
        model.eval()
    except ModelNotFoundError:
        raise
    except Exception as e:
        raise ModelLoadError(str(e))
    _cache.clear()
    _cache[key] = (model, ckpt)
    return _cache[key]
