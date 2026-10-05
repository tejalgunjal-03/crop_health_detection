"""Inspect the maize LEAF dataset and write a report. Nothing is hard-coded:
classes are discovered from folder names, counts from the files on disk."""
import argparse, hashlib, json, random
from collections import Counter, defaultdict
from pathlib import Path
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

def find_root(base: Path) -> Path:
    """Return the folder whose sub-folders are the class folders."""
    for cand in [base, *sorted(p for p in base.rglob("*") if p.is_dir())]:
        subs = [d for d in cand.iterdir() if d.is_dir()]
        if len(subs) >= 2 and all(any(f.suffix.lower() in EXTS for f in s.iterdir()) for s in subs):
            return cand
    raise SystemExit(f"No class folders found under {base}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="datasets/leaf")
    ap.add_argument("--out", default="results/reports")
    a = ap.parse_args()
    root = find_root(Path(a.data)); out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    classes = sorted(d.name for d in root.iterdir() if d.is_dir())
    per_class, fmts, sizes, modes = Counter(), Counter(), Counter(), Counter()
    corrupt, hashes, dup = [], defaultdict(list), []
    files_by_class = defaultdict(list); non_image = []
    for c in classes:
        for f in sorted((root / c).iterdir()):
            if f.suffix.lower() not in EXTS:
                non_image.append(str(f)); continue
            try:
                with Image.open(f) as im:
                    im.verify()
                with Image.open(f) as im:
                    sizes[im.size] += 1; modes[im.mode] += 1; fmts[im.format] += 1
            except Exception as e:
                corrupt.append(f"{f}: {e}"); continue
            per_class[c] += 1; files_by_class[c].append(f)
            hashes[hashlib.md5(f.read_bytes()).hexdigest()].append(str(f))
    dup = [v for v in hashes.values() if len(v) > 1]
    total = sum(per_class.values())
    mx, mn = max(per_class.values()), min(per_class.values())
    rep = dict(root=str(root), classes=classes, total_valid=total, per_class=dict(per_class),
               formats=dict(fmts), color_modes=dict(modes),
               n_distinct_sizes=len(sizes), top_sizes={f"{k[0]}x{k[1]}": v for k, v in sizes.most_common(5)},
               corrupt=corrupt, non_image_files=non_image, exact_duplicate_groups=len(dup),
               duplicate_examples=dup[:5], imbalance_ratio_max_over_min=round(mx / mn, 3))
    (out / "leaf_dataset_report.json").write_text(json.dumps(rep, indent=2))
    lines = ["LEAF DATASET REPORT", "=" * 40, f"Root: {root}", f"Valid images: {total}",
             "Per class:"] + [f"  {c}: {per_class[c]} ({100*per_class[c]/total:.1f}%)" for c in classes] + [
             f"Formats: {dict(fmts)}", f"Color modes: {dict(modes)}",
             f"Distinct image sizes: {len(sizes)}; most common: {rep['top_sizes']}",
             f"Corrupt: {len(corrupt)}", f"Non-image files: {len(non_image)}",
             f"Exact duplicate groups (md5): {len(dup)}",
             f"Imbalance (largest/smallest class): {rep['imbalance_ratio_max_over_min']}"]
    (out / "leaf_dataset_report.txt").write_text("\n".join(lines)); print("\n".join(lines))
    random.seed(0)
    fig, ax = plt.subplots(len(classes), 4, figsize=(10, 2.6 * len(classes)))
    for i, c in enumerate(classes):
        for j, f in enumerate(random.sample(files_by_class[c], min(4, len(files_by_class[c])))):
            ax[i][j].imshow(Image.open(f).convert("RGB")); ax[i][j].axis("off")
            if j == 0: ax[i][j].set_title(c, fontsize=9, loc="left")
    plt.tight_layout(); plt.savefig(out / "leaf_samples.png", dpi=80)
    print(f"\nSaved reports to {out}")

if __name__ == "__main__":
    main()
