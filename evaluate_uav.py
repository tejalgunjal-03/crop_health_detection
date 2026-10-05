import sys
sys.path.insert(0, "scripts")

import json
from pathlib import Path

import numpy as np
import torch
import yaml
import train_uav as t

cfg = yaml.safe_load(open("configs/uav_classes.yaml"))
names = [cfg["classes"][k]["name"] for k in sorted(cfg["classes"])]

stems, X, Y, groups = t.load_data(Path("datasets/uav"))

with open("results/reports/uav_splits.json") as f:
    splits = json.load(f)

idx = {v: i for i, v in enumerate(stems)}
te_i = np.array([idx[x] for x in splits["test"]])

te = torch.utils.data.DataLoader(
    t.SegDS(X[te_i], Y[te_i], False),
    batch_size=4,
    shuffle=False,
    num_workers=0,
)

ck = torch.load(
    "models/uav/best_model.pth",
    map_location="cpu",
    weights_only=True
)

model = t.UNet(
    ck["num_classes"],
    base=ck["base_channels"]
)

model.load_state_dict(ck["model_state"])

test = t.metrics_from_cm(
    t.confusion(model, te, ck["num_classes"]),
    names
)

Path("results/reports/uav_test_metrics.json").write_text(
    json.dumps(test, indent=2)
)

print()
print(
    f"TEST: mIoU={test['mean_iou']:.4f} "
    f"mDice={test['mean_dice']:.4f} "
    f"pixel_acc={test['pixel_accuracy']:.4f}"
)

for c, m in test["per_class"].items():
    def fmt(v):
        return "n/a" if v is None else f"{v:.3f}"

    print(
        f"  {c:20s} "
        f"IoU={fmt(m['iou'])} "
        f"Dice={fmt(m['dice'])} "
        f"P={fmt(m['precision'])} "
        f"R={fmt(m['recall'])}"
    )

print()
print("Saved: results/reports/uav_test_metrics.json")
