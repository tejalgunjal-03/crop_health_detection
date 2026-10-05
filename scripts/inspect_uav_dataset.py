"""Inspect the UAV RGB + PNG-mask dataset. Mask class IDs are DISCOVERED here,
never assumed. The 6-channel .npy files are intentionally never opened."""
import argparse, csv, json, random
from collections import Counter
from pathlib import Path
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def find_root(base: Path) -> Path:
    for p in [base, *base.rglob("*")]:
        if p.is_dir() and (p / "images").is_dir() and (p / "masks").is_dir():
            return p
    raise SystemExit(f"No images/ + masks/ folders under {base}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="datasets/uav"); ap.add_argument("--out", default="results/reports")
    a = ap.parse_args(); root = find_root(Path(a.data)); out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    imgs = {p.stem: p for p in (root / "images").glob("*.jpg")}
    masks = {p.stem: p for p in (root / "masks").glob("*.png")}
    stems = sorted(set(imgs) & set(masks))
    missing_masks = sorted(set(imgs) - set(masks)); missing_imgs = sorted(set(masks) - set(imgs))
    corrupt, img_sizes, img_modes, mask_sizes, mask_modes = [], Counter(), Counter(), Counter(), Counter()
    val_pixels, val_images, per_group = Counter(), Counter(), Counter()
    group_vals = {}
    manifest = {}
    mf = root / "meta" / "subset_manifest.csv"
    if mf.exists():
        with open(mf, newline="") as f:
            manifest = {r["stem"]: r["group"] for r in csv.DictReader(f)}
    for s in stems:
        try:
            with Image.open(imgs[s]) as im:
                im.load(); img_sizes[im.size] += 1; img_modes[im.mode] += 1
            with Image.open(masks[s]) as m:
                m.load(); mask_sizes[m.size] += 1; mask_modes[m.mode] += 1; arr = np.array(m)
        except Exception as e:
            corrupt.append(f"{s}: {e}"); continue
        u, c = np.unique(arr, return_counts=True)
        g = manifest.get(s, "unknown"); per_group[g] += 1
        gv = group_vals.setdefault(g, Counter())
        for v, n in zip(u.tolist(), c.tolist()):
            val_pixels[v] += n; val_images[v] += 1; gv[v] += n
    tot = sum(val_pixels.values())
    rep = dict(root=str(root), n_rgb=len(imgs), n_masks=len(masks), n_matched=len(stems),
               missing_masks=missing_masks, missing_images=missing_imgs, corrupt=corrupt,
               rgb_sizes={f"{k[0]}x{k[1]}": v for k, v in img_sizes.items()}, rgb_modes=dict(img_modes),
               mask_sizes={f"{k[0]}x{k[1]}": v for k, v in mask_sizes.items()}, mask_modes=dict(mask_modes),
               mask_unique_values=sorted(val_pixels), pixel_count_per_value=dict(sorted(val_pixels.items())),
               pixel_fraction_per_value={k: round(v / tot, 5) for k, v in sorted(val_pixels.items())},
               images_containing_value=dict(sorted(val_images.items())),
               groups_in_manifest=dict(per_group),
               values_by_group={g: dict(sorted(c.items())) for g, c in group_vals.items()},
               npy_files_present_but_unused=len(list((root / "images").glob("*.npy"))))
    (out / "uav_dataset_report.json").write_text(json.dumps(rep, indent=2))
    txt = ["UAV DATASET REPORT", "=" * 40, f"Root: {root}", f"RGB jpg: {len(imgs)} | PNG masks: {len(masks)} | matched: {len(stems)}",
           f"Missing masks: {len(missing_masks)} | Missing images: {len(missing_imgs)} | Corrupt: {len(corrupt)}",
           f"RGB sizes: {rep['rgb_sizes']} modes: {dict(img_modes)}", f"Mask sizes: {rep['mask_sizes']} modes: {dict(mask_modes)}",
           f"Mask unique values: {rep['mask_unique_values']}", "Pixel fraction per value: " + str(rep['pixel_fraction_per_value']),
           "Images containing value: " + str(rep['images_containing_value']), f"Manifest groups: {dict(per_group)}",
           "Values by group (pixel counts): " + str(rep['values_by_group']),
           "NOTE: the README does NOT define the numeric class IDs. Map values to names yourself\n"
           "after looking at masks_viz/ and the per-group table above; record it in configs/uav_classes.yaml."]
    (out / "uav_dataset_report.txt").write_text("\n".join(txt)); print("\n".join(txt))
    random.seed(0); pick = random.sample(stems, min(6, len(stems)))
    fig, ax = plt.subplots(len(pick), 2, figsize=(5, 2.5 * len(pick)))
    vmax = max(val_pixels) if val_pixels else 1
    for i, s in enumerate(pick):
        ax[i][0].imshow(Image.open(imgs[s]).convert("RGB")); ax[i][0].set_title(manifest.get(s, ""), fontsize=7)
        ax[i][1].imshow(np.array(Image.open(masks[s])), vmin=0, vmax=vmax, cmap="tab10", interpolation="nearest")
        ax[i][0].axis("off"); ax[i][1].axis("off")
    plt.tight_layout(); plt.savefig(out / "uav_samples.png", dpi=80)

if __name__ == "__main__":
    main()
