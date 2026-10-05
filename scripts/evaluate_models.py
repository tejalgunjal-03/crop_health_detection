"""Re-evaluate saved checkpoints on the SAVED test splits and write a summary.
Run:  python scripts/evaluate_models.py"""
import json, sys
from pathlib import Path
import torch
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend")); sys.path.insert(0, str(ROOT / "scripts"))
REP = ROOT / "results/reports"; summary = {}

leaf_ck = ROOT / "models/leaf/best_model.pth"
if leaf_ck.exists() and (REP / "leaf_splits.json").exists():
    from train_leaf import evaluate_split, find_root
    from app.ml.leaf.model import build_leaf_model
    ck = torch.load(leaf_ck, map_location="cpu", weights_only=True)
    m = build_leaf_model(len(ck["class_names"]), pretrained=False); m.load_state_dict(ck["model_state"])
    sp = json.loads((REP / "leaf_splits.json").read_text())
    summary["leaf_test"] = evaluate_split(m, find_root(ROOT / "datasets/leaf"), sp["test"], ck["class_names"], ck["img_size"], ck["mean"], ck["std"])
else:
    summary["leaf_test"] = "Trained model not found. Please train the model first (python scripts/train_leaf.py)."

uav_ck = ROOT / "models/uav/best_model.pth"
if uav_ck.exists() and (REP / "uav_splits.json").exists():
    from train_uav import load_data, SegDS, confusion, metrics_from_cm
    from app.ml.uav.unet import UNet
    ck = torch.load(uav_ck, map_location="cpu", weights_only=True)
    m = UNet(ck["num_classes"], base=ck["base_channels"]); m.load_state_dict(ck["model_state"])
    stems, X, Y, _ = load_data(ROOT / "datasets/uav"); pos = {s: i for i, s in enumerate(stems)}
    ii = [pos[s] for s in json.loads((REP / "uav_splits.json").read_text())["test"]]
    dl = torch.utils.data.DataLoader(SegDS(X[ii], Y[ii], False), batch_size=4)
    summary["uav_test"] = metrics_from_cm(confusion(m, dl, ck["num_classes"]), ck["class_names"])
else:
    summary["uav_test"] = "Trained model not found. Please train the model first (python scripts/train_uav.py)."

(REP / "evaluation_summary.json").write_text(json.dumps(summary, indent=2))
for k, v in summary.items():
    print(k, "->", v if isinstance(v, str) else {x: v[x] for x in v if not isinstance(v[x], (dict, list))})
