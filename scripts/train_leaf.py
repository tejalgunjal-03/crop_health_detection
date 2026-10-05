"""Train the maize LEAF classifier (MobileNetV3-Small, transfer learning, CPU).
Run:  python scripts/train_leaf.py --epochs 12
Training is ALWAYS separate from the API. All reported numbers are measured here."""
from __future__ import annotations
import argparse, hashlib, json, random, sys, time
from pathlib import Path
import numpy as np
import torch, torch.nn as nn
from PIL import Image
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend")); sys.path.insert(0, str(ROOT / "scripts"))
from app.ml.leaf.model import build_leaf_model, eval_transform, train_transform, IMAGENET_MEAN, IMAGENET_STD
from inspect_leaf_dataset import find_root, EXTS


class LeafDS(torch.utils.data.Dataset):
    def __init__(self, items, tf): self.items, self.tf = items, tf
    def __len__(self): return len(self.items)
    def __getitem__(self, i):
        p, y = self.items[i]
        with Image.open(p) as im:
            return self.tf(im.convert("RGB")), y        # handles RGBA / CMYK files too


def build_splits(data_dir: Path, seed: int):
    """Stratified 70/15/15. Exact byte-duplicates are kept out of the splits (files are NOT deleted)."""
    root = find_root(data_dir)
    classes = sorted(d.name for d in root.iterdir() if d.is_dir())
    items, seen, dropped = [], {}, []
    for ci, c in enumerate(classes):
        for f in sorted((root / c).iterdir()):
            if f.suffix.lower() not in EXTS: continue
            h = hashlib.md5(f.read_bytes()).hexdigest()
            if h in seen: dropped.append(str(f.relative_to(root))); continue
            seen[h] = 1; items.append((str(f.relative_to(root)), ci))
    paths, labels = zip(*items)
    p_tr, p_rest, y_tr, y_rest = train_test_split(paths, labels, test_size=0.30, stratify=labels, random_state=seed)
    p_va, p_te, y_va, y_te = train_test_split(p_rest, y_rest, test_size=0.50, stratify=y_rest, random_state=seed)
    mk = lambda P, Y: [[p, int(y)] for p, y in zip(P, Y)]
    return root, classes, {"train": mk(p_tr, y_tr), "val": mk(p_va, y_va), "test": mk(p_te, y_te)}, dropped


@torch.no_grad()
def predict_all(model, loader):
    model.eval(); ys, ps = [], []
    for x, y in loader:
        ps += model(x).argmax(1).tolist(); ys += y.tolist()
    return np.array(ys), np.array(ps)


def evaluate_split(model, root, items, classes, size, mean, std, batch=32):
    dl = torch.utils.data.DataLoader(LeafDS([(root / p, y) for p, y in items], eval_transform(size, mean, std)),
                                     batch_size=batch, shuffle=False, num_workers=0)
    y, p = predict_all(model, dl)
    rep = classification_report(y, p, labels=list(range(len(classes))), target_names=classes, output_dict=True, zero_division=0)
    return {"n": int(len(y)), "accuracy": float(accuracy_score(y, p)),
            "macro_precision": rep["macro avg"]["precision"], "macro_recall": rep["macro avg"]["recall"],
            "macro_f1": rep["macro avg"]["f1-score"], "per_class": {c: rep[c] for c in classes},
            "confusion_matrix": confusion_matrix(y, p, labels=list(range(len(classes)))).tolist()}


def save_confusion_png(cm, classes, path):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(5, 4.5)); ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(classes)), classes, rotation=30, ha="right"); ax.set_yticks(range(len(classes)), classes)
    for i in range(len(classes)):
        for j in range(len(classes)): ax.text(j, i, cm[i][j], ha="center", va="center")
    ax.set_xlabel("Predicted"); ax.set_ylabel("True"); plt.tight_layout(); plt.savefig(path, dpi=100); plt.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(ROOT / "datasets/leaf")); ap.add_argument("--out", default=str(ROOT / "models/leaf"))
    ap.add_argument("--reports", default=str(ROOT / "results/reports"))
    ap.add_argument("--epochs", type=int, default=12); ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3); ap.add_argument("--img-size", type=int, default=224)
    ap.add_argument("--seed", type=int, default=42); ap.add_argument("--patience", type=int, default=4)
    ap.add_argument("--workers", type=int, default=0); ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--no-pretrained", action="store_true", help="use if you are offline (lower accuracy expected)")
    a = ap.parse_args()
    random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed)
    if a.threads: torch.set_num_threads(a.threads)
    out, rep_dir = Path(a.out), Path(a.reports); out.mkdir(parents=True, exist_ok=True); rep_dir.mkdir(parents=True, exist_ok=True)

    root, classes, splits, dropped = build_splits(Path(a.data), a.seed)
    (rep_dir / "leaf_splits.json").write_text(json.dumps({"classes": classes, "seed": a.seed, "dropped_exact_duplicates": dropped, **splits}))
    print(f"classes={classes}  train/val/test={[len(splits[k]) for k in ('train','val','test')]}  duplicates excluded={len(dropped)}")

    mk = lambda k, tf, sh: torch.utils.data.DataLoader(LeafDS([(root / p, y) for p, y in splits[k]], tf),
                                                       batch_size=a.batch_size, shuffle=sh, num_workers=a.workers)
    tr = mk("train", train_transform(a.img_size), True); va = mk("val", eval_transform(a.img_size), False)

    model = build_leaf_model(len(classes), pretrained=not a.no_pretrained)
    counts = np.bincount([y for _, y in splits["train"]], minlength=len(classes))
    weights = torch.tensor(counts.sum() / (len(classes) * counts), dtype=torch.float32)   # class-imbalance weights
    crit = nn.CrossEntropyLoss(weight=weights)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=a.epochs)

    hist, best_f1, bad = [], -1.0, 0
    ckpt_path = out / "best_model.pth"
    for ep in range(1, a.epochs + 1):
        t0 = time.time(); model.train(); tl = n = 0
        for x, y in tr:
            opt.zero_grad(); loss = crit(model(x), y); loss.backward(); opt.step()
            tl += loss.item() * len(y); n += len(y)
        sched.step()
        yv, pv = predict_all(model, va)
        vacc, vf1 = accuracy_score(yv, pv), f1_score(yv, pv, average="macro")
        hist.append({"epoch": ep, "train_loss": tl / n, "val_accuracy": vacc, "val_macro_f1": vf1, "seconds": time.time() - t0})
        print(f"epoch {ep}/{a.epochs}  loss {tl/n:.4f}  val_acc {vacc:.4f}  val_f1 {vf1:.4f}  ({time.time()-t0:.0f}s)", flush=True)
        if vf1 > best_f1:
            best_f1, bad = vf1, 0
            torch.save({"model_state": model.state_dict(), "class_names": classes, "img_size": a.img_size,
                        "mean": IMAGENET_MEAN, "std": IMAGENET_STD, "arch": "mobilenet_v3_small",
                        "pretrained": not a.no_pretrained, "seed": a.seed, "best_epoch": ep, "val_macro_f1": vf1}, ckpt_path)
        else:
            bad += 1
            if bad >= a.patience: print("early stopping"); break
    (rep_dir / "leaf_training_history.json").write_text(json.dumps(hist, indent=2))

    ck = torch.load(ckpt_path, map_location="cpu", weights_only=True); model.load_state_dict(ck["model_state"])
    test = evaluate_split(model, root, splits["test"], classes, a.img_size, IMAGENET_MEAN, IMAGENET_STD)
    (rep_dir / "leaf_test_metrics.json").write_text(json.dumps(test, indent=2))
    save_confusion_png(test["confusion_matrix"], classes, rep_dir / "leaf_confusion_matrix.png")
    print(f"\nTEST (n={test['n']}): accuracy={test['accuracy']:.4f} macro_f1={test['macro_f1']:.4f}")
    for c, m in test["per_class"].items(): print(f"  {c:16s} P={m['precision']:.3f} R={m['recall']:.3f} F1={m['f1-score']:.3f} n={int(m['support'])}")
    print(f"Saved: {ckpt_path}")


if __name__ == "__main__":
    main()
