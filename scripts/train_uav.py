"""Train the UAV RGB-only U-Net on 224x224 patches + PNG masks (CPU).
Run:  python scripts/train_uav.py --epochs 30
The 6-channel .npy files are never read."""
from __future__ import annotations
import argparse, csv, json, random, sys, time
from pathlib import Path
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
from PIL import Image
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend")); sys.path.insert(0, str(ROOT / "scripts"))
from app.ml.uav.unet import UNet
from inspect_uav_dataset import find_root


def load_data(data_dir: Path):
    root = find_root(data_dir)
    stems = sorted(p.stem for p in (root / "images").glob("*.jpg") if (root / "masks" / f"{p.stem}.png").exists())
    X = np.stack([np.array(Image.open(root / "images" / f"{s}.jpg").convert("RGB")) for s in stems])
    Y = np.stack([np.array(Image.open(root / "masks" / f"{s}.png")) for s in stems])
    groups = {}
    mf = root / "meta" / "subset_manifest.csv"
    if mf.exists():
        with open(mf, newline="") as f: groups = {r["stem"]: r["group"] for r in csv.DictReader(f)}
    g = [groups.get(s, "unknown") for s in stems]
    return stems, X, Y, g


class SegDS(torch.utils.data.Dataset):
    def __init__(self, X, Y, augment): self.X, self.Y, self.aug = X, Y, augment
    def __len__(self): return len(self.X)
    def __getitem__(self, i):
        x, y = self.X[i].copy(), self.Y[i].copy()
        if self.aug:
            # Flips and rotation
            if random.random() < .5: x, y = x[:, ::-1], y[:, ::-1]
            if random.random() < .5: x, y = x[::-1], y[::-1]
            k = random.randint(0, 3); x, y = np.rot90(x, k), np.rot90(y, k)
            x = np.ascontiguousarray(x).astype(np.float32)
            # Colour jitter: brightness, contrast, saturation
            x = x * random.uniform(0.75, 1.25)                   # brightness
            mean = x.mean(axis=(0, 1), keepdims=True)
            x = mean + (x - mean) * random.uniform(0.75, 1.25)   # contrast
            if random.random() < 0.5:                             # channel shuffle (hue-like)
                ch = [0, 1, 2]; random.shuffle(ch); x = x[:, :, ch]
            # Gaussian blur
            if random.random() < 0.3:
                import cv2 as _cv; sig = random.choice([1, 1, 3])
                x = _cv.GaussianBlur(x.astype(np.uint8), (0, 0), sig).astype(np.float32)
            x = np.clip(x, 0, 255)
            # Random Cutout
            H, W = x.shape[:2]
            for _ in range(random.randint(0, 3)):
                cw = random.randint(20, 50); ch2 = random.randint(20, 50)
                cx = random.randint(0, max(0, W - cw)); cy = random.randint(0, max(0, H - ch2))
                x[cy:cy+ch2, cx:cx+cw] = x.mean()   # fill with mean colour
                
            # Random scale (zoom in/out) to simulate different flight altitudes
            if random.random() < 0.5:
                import cv2 as _cv
                scale = random.uniform(0.5, 1.5)
                if scale < 1.0:
                    ns = int(224 * scale)
                    xr = _cv.resize(x, (ns, ns), interpolation=_cv.INTER_LINEAR)
                    yr = _cv.resize(y.astype(np.uint8), (ns, ns), interpolation=_cv.INTER_NEAREST)
                    pad = 224 - ns; top = random.randint(0, pad); bottom = pad - top; left = random.randint(0, pad); right = pad - left
                    x = _cv.copyMakeBorder(xr, top, bottom, left, right, _cv.BORDER_REFLECT_101)
                    y = _cv.copyMakeBorder(yr, top, bottom, left, right, _cv.BORDER_REFLECT_101)
                else:
                    ns = int(224 * scale)
                    xr = _cv.resize(x, (ns, ns), interpolation=_cv.INTER_LINEAR)
                    yr = _cv.resize(y.astype(np.uint8), (ns, ns), interpolation=_cv.INTER_NEAREST)
                    top = random.randint(0, ns - 224); left = random.randint(0, ns - 224)
                    x = xr[top:top+224, left:left+224]
                    y = yr[top:top+224, left:left+224]
        else:
            x = np.ascontiguousarray(x).astype(np.float32)
        t = torch.from_numpy(np.ascontiguousarray(x)).permute(2, 0, 1).float() / 255.0
        return t, torch.from_numpy(np.ascontiguousarray(y)).long()


def dice_loss(logits, y, n):
    p = F.softmax(logits, 1); oh = F.one_hot(y, n).permute(0, 3, 1, 2).float()
    inter = (p * oh).sum((0, 2, 3)); den = p.sum((0, 2, 3)) + oh.sum((0, 2, 3))
    # Removing [present] filter so we penalize false positives on empty classes!
    return 1 - ((2 * inter + 1) / (den + 1)).mean()


@torch.no_grad()
def confusion(model, loader, n):
    model.eval(); cm = np.zeros((n, n), np.int64)
    for x, y in loader:
        p = model(x).argmax(1)
        cm += np.bincount((y * n + p).flatten().numpy(), minlength=n * n).reshape(n, n)
    return cm


def metrics_from_cm(cm, names):
    tp = np.diag(cm).astype(float); fp = cm.sum(0) - tp; fn = cm.sum(1) - tp
    with np.errstate(divide="ignore", invalid="ignore"):
        iou, dice = tp / (tp + fp + fn), 2 * tp / (2 * tp + fp + fn)
        prec, rec = tp / (tp + fp), tp / (tp + fn)
    nz = lambda v: None if np.isnan(v) else float(v)
    return {"pixel_accuracy": float(tp.sum() / cm.sum()), "mean_iou": float(np.nanmean(iou)), "mean_dice": float(np.nanmean(dice)),
            "per_class": {n: {"iou": nz(iou[i]), "dice": nz(dice[i]), "precision": nz(prec[i]), "recall": nz(rec[i]),
                              "pixels_true": int(cm[i].sum())} for i, n in enumerate(names)},
            "confusion_matrix": cm.tolist()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(ROOT / "datasets/uav")); ap.add_argument("--out", default=str(ROOT / "models/uav"))
    ap.add_argument("--reports", default=str(ROOT / "results/reports"))
    ap.add_argument("--epochs", type=int, default=50); ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--lr", type=float, default=2e-3); ap.add_argument("--base", type=int, default=16)
    ap.add_argument("--seed", type=int, default=42); ap.add_argument("--patience", type=int, default=10)
    ap.add_argument("--threads", type=int, default=0)
    a = ap.parse_args()
    random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed)
    if a.threads: torch.set_num_threads(a.threads)
    out, rep = Path(a.out), Path(a.reports); out.mkdir(parents=True, exist_ok=True); rep.mkdir(parents=True, exist_ok=True)

    import yaml
    cfg = yaml.safe_load(open(ROOT / "configs/uav_classes.yaml")); names = [cfg["classes"][k]["name"] for k in sorted(cfg["classes"])]
    stems, X, Y, groups = load_data(Path(a.data))
    n = len(names); seen = np.unique(Y).tolist()
    if max(seen) >= n: raise SystemExit(f"Mask values {seen} exceed the {n} classes in configs/uav_classes.yaml. Re-inspect the dataset.")
    print(f"{len(stems)} patches, mask values present={seen}, classes={names}")

    idx = np.arange(len(stems))
    tr_i, rest = train_test_split(idx, test_size=.30, stratify=groups, random_state=a.seed)
    va_i, te_i = train_test_split(rest, test_size=.50, stratify=[groups[i] for i in rest], random_state=a.seed)
    (rep / "uav_splits.json").write_text(json.dumps({k: [stems[i] for i in v] for k, v in (("train", tr_i), ("val", va_i), ("test", te_i))}))
    print(f"train/val/test = {len(tr_i)}/{len(va_i)}/{len(te_i)}  (stratified by water/rust group)")
    mk = lambda ii, aug, sh: torch.utils.data.DataLoader(SegDS(X[ii], Y[ii], aug), batch_size=a.batch_size, shuffle=sh, num_workers=0)
    tr, va, te = mk(tr_i, True, True), mk(va_i, False, False), mk(te_i, False, False)

    freq = np.bincount(Y[tr_i].ravel(), minlength=n).astype(float) + 1
    # Higher exponent (0.75) penalises minority classes more — helps rust, water stress recall
    w = torch.tensor((freq.sum() / freq) ** 0.75, dtype=torch.float32); w /= w.mean()
    ce = nn.CrossEntropyLoss(weight=w)
    model = UNet(n, base=a.base)
    print(f"U-Net parameters: {sum(p.numel() for p in model.parameters()):,}")
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=.5, patience=3)

    hist, best, bad, path = [], -1.0, 0, out / "best_model.pth"
    for ep in range(1, a.epochs + 1):
        t0 = time.time(); model.train(); tl = k = 0
        for x, y in tr:
            opt.zero_grad(); lg = model(x); loss = ce(lg, y) + dice_loss(lg, y, n); loss.backward(); opt.step()
            tl += loss.item() * len(y); k += len(y)
        vm = metrics_from_cm(confusion(model, va, n), names); sched.step(1 - vm["mean_iou"])
        hist.append({"epoch": ep, "train_loss": tl / k, "val_mean_iou": vm["mean_iou"], "val_mean_dice": vm["mean_dice"],
                     "val_pixel_accuracy": vm["pixel_accuracy"], "seconds": time.time() - t0})
        print(f"epoch {ep}/{a.epochs} loss {tl/k:.4f} val_mIoU {vm['mean_iou']:.4f} val_pixAcc {vm['pixel_accuracy']:.4f} ({time.time()-t0:.0f}s)", flush=True)
        if vm["mean_iou"] > best:
            best, bad = vm["mean_iou"], 0
            torch.save({"model_state": model.state_dict(), "num_classes": n, "class_names": names, "base_channels": a.base,
                        "tile_size": 224, "seed": a.seed, "best_epoch": ep, "val_mean_iou": best}, path)
        else:
            bad += 1
            if bad >= a.patience: print("early stopping"); break
    (rep / "uav_training_history.json").write_text(json.dumps(hist, indent=2))

    model.load_state_dict(torch.load(path, map_location="cpu", weights_only=True)["model_state"])
    test = metrics_from_cm(confusion(model, te, n), names)
    (rep / "uav_test_metrics.json").write_text(json.dumps(test, indent=2))
    print(f"\nTEST: mIoU={test['mean_iou']:.4f} mDice={test['mean_dice']:.4f} pixel_acc={test['pixel_accuracy']:.4f}")
    for c, m in test["per_class"].items():
        f = lambda v: "n/a" if v is None else f"{v:.3f}"
        print(f"  {c:20s} IoU={f(m['iou'])} Dice={f(m['dice'])} P={f(m['precision'])} R={f(m['recall'])}")
    print(f"Saved: {path}")


if __name__ == "__main__":
    main()
