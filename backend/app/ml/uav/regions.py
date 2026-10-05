"""Turn a predicted label map into suspicious-region boxes and a coloured overlay."""
from __future__ import annotations
import cv2
import numpy as np


def class_fractions(labels: np.ndarray, classes: dict) -> dict:
    total = labels.size
    return {classes[k]["name"]: round(float((labels == k).sum()) / total, 5) for k in classes}


def find_regions(labels: np.ndarray, suspicious_ids: list[int], classes: dict,
                 min_area_px: int, max_regions: int = 20) -> list[dict]:
    out = []
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    
    for cid in suspicious_ids:
        if cid not in classes:
            continue
        mask = (labels == cid).astype(np.uint8) * 255

        n, cc, stats, cent = cv2.connectedComponentsWithStats(mask, connectivity=8)
        
        for i in range(1, n):
            x, y, w, h, area = stats[i]
            if area < min_area_px:
                continue
                
            # The dominant class for this region is the one that triggered it
            dom = cid
            
            # Use ALL pixels inside the bounding box to reflect the area's composition
            inside = labels[y:y + h, x:x + w].ravel()
            total = inside.size
            ids, counts = np.unique(inside, return_counts=True)
            
            # Build per-class composition breakdown
            composition = [
                {
                    "class_name": classes[int(c_id)]["name"],
                    "label": classes[int(c_id)]["label"],
                    "color": classes[int(c_id)]["color"],
                    "pixel_count": int(cnt),
                    "fraction": round(float(cnt) / total, 4),
                    "is_suspicious": int(c_id) in suspicious_ids,
                }
                for c_id, cnt in zip(ids.tolist(), counts.tolist())
                if int(c_id) in classes
            ]
            composition.sort(key=lambda c: c["fraction"], reverse=True)
            
            out.append({"x": int(x), "y": int(y), "width": int(w), "height": int(h), "area_px": int(area),
                        "dominant_class": classes[dom]["name"], "dominant_label": classes[dom]["label"],
                        "color": classes[dom]["color"], "composition": composition})
                        
    out.sort(key=lambda r: r["area_px"], reverse=True)
    for i, r in enumerate(out[:max_regions], 1):
        r["id"] = i
    return out[:max_regions]


def make_overlay(rgb: np.ndarray, labels: np.ndarray, classes: dict, suspicious_ids: list[int],
                 regions: list[dict], alpha: float = 0.45, max_side: int = 2000) -> np.ndarray:
    # Build the full class-colour layer for every pixel
    colour = np.zeros_like(rgb)
    for k, v in classes.items():
        colour[labels == k] = v["color"]

    # Blend the FULL segmentation map over the entire image so every class
    # (soil, healthy, water stress, rust) is visible everywhere — not just inside boxes.
    out = (rgb * (1 - alpha) + colour * alpha).astype(np.uint8)

    for r in regions:
        dom_color = tuple(int(c) for c in r.get("color", (0, 255, 255)))
        cv2.rectangle(out, (r["x"], r["y"]), (r["x"] + r["width"], r["y"] + r["height"]), dom_color, 4)

        # Label: "R1: Rust 68% | Healthy 22%" — top-2 classes by fraction
        top2 = r.get("composition", [])[:2]
        label_parts = [f"{c['label']} {int(c['fraction'] * 100)}%" for c in top2]
        text = f"R{r['id']}: " + " | ".join(label_parts)

        font, scale, thickness = cv2.FONT_HERSHEY_SIMPLEX, 0.65, 2
        (tw, th), baseline = cv2.getTextSize(text, font, scale, thickness)
        tx, ty = r["x"], r["y"]
        # Clamp text to stay inside the image
        ty = max(ty, th + baseline + 4)
        # Dark background pill for readability
        cv2.rectangle(out, (tx, ty - th - baseline - 4), (tx + tw + 8, ty + baseline), (0, 0, 0), -1)
        cv2.putText(out, text, (tx + 4, ty - baseline), font, scale, dom_color, thickness)

    s = max(out.shape[:2])
    if s > max_side:
        f = max_side / s
        out = cv2.resize(out, None, fx=f, fy=f, interpolation=cv2.INTER_AREA)
    return out
