
import argparse
import shutil
from pathlib import Path
import random

import cv2
import numpy as np

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}



def _gaussian_noise(img, sigma):
    noise = np.random.normal(0, sigma, img.shape).astype(np.float32)
    out = img.astype(np.float32) + noise
    return np.clip(out, 0, 255).astype(np.uint8)

def _motion_blur(img, ksize):
    k = max(1, int(ksize))
    kernel = np.zeros((k, k), dtype=np.float32)
    kernel[k // 2, :] = 1.0 / k
    return cv2.filter2D(img, -1, kernel)

def _gaussian_blur(img, ksize):
    k = max(1, int(ksize))
    if k % 2 == 0:
        k += 1
    return cv2.GaussianBlur(img, (k, k), 0)

def _low_light(img, alpha, beta):
    out = img.astype(np.float32) * alpha + beta
    return np.clip(out, 0, 255).astype(np.uint8)

def _occlusion(img, frac_h, frac_w):
    h, w = img.shape[:2]
    oh = max(1, int(h * frac_h))
    ow = max(1, int(w * frac_w))
    y0 = np.random.randint(0, max(1, h - oh + 1))
    x0 = np.random.randint(0, max(1, w - ow + 1))
    out = img.copy()
    out[y0:y0 + oh, x0:x0 + ow] = 0
    return out

def _params_from_severity(sev: int):
    s = int(np.clip(sev, 1, 5))
    return {
        "sigma": 5 * s,            # Gaussian noise std
        "mblur": 1 + 2 * s,        # Motion blur kernel
        "gblur": 1 + 2 * s,        # Gaussian blur kernel
        "alpha": 1.0 - 0.08 * s,   # brightness multiplier
        "beta":  -4 * s,           # brightness shift
        "occ_h": 0.02 * s,         # occlusion fractions
        "occ_w": 0.02 * s,
        "p_noise": 0.7,
        "p_mblur": 0.5,
        "p_gblur": 0.4,
        "p_low":   0.6,
        "p_occ":   0.3,
    }

def _apply_pipeline(img_bgr, params, rng):
    img = img_bgr
    if rng.random() < params["p_noise"]:
        img = _gaussian_noise(img, params["sigma"])

    r = rng.random()
    if r < params["p_mblur"]:
        img = _motion_blur(img, params["mblur"])
    elif r < params["p_mblur"] + params["p_gblur"]:
        img = _gaussian_blur(img, params["gblur"])

    if rng.random() < params["p_low"]:
        img = _low_light(img, params["alpha"], params["beta"])

    if rng.random() < params["p_occ"]:
        img = _occlusion(img, params["occ_h"], params["occ_w"])

    return img
# ----------------------------------------------------------------------------


def add_noise_yolo(in_root, out_root, severity=2, seed=42):
    rng = random.Random(seed)
    np.random.seed(seed)

    in_root = Path(in_root)
    out_root = Path(out_root)
    in_imgs = in_root / "images"
    in_labs = in_root / "labels"

    if not in_imgs.is_dir() or not in_labs.is_dir():
        raise FileNotFoundError(f"Expected YOLO layout: {in_root}/images and {in_root}/labels")

    out_imgs = out_root / "images"
    out_labs = out_root / "labels"
    out_imgs.mkdir(parents=True, exist_ok=True)
    out_labs.mkdir(parents=True, exist_ok=True)

    params = _params_from_severity(severity)

    img_paths = [p for p in in_imgs.rglob("*") if p.suffix.lower() in IMG_EXTS and p.is_file()]
    img_paths.sort()
    print(f"Found {len(img_paths)} images in {in_imgs}")

    written = 0
    for i, ipath in enumerate(img_paths, 1):
        # read image
        img = cv2.imread(str(ipath), cv2.IMREAD_COLOR)
        if img is None:
            print(f"[warn] cannot read {ipath}")
            continue

        # distort
        out_img = _apply_pipeline(img, params, rng)

        # destination path (preserve subfolders)
        rel = ipath.relative_to(in_imgs)
        opath = out_imgs / rel
        opath.parent.mkdir(parents=True, exist_ok=True)

        ok = cv2.imwrite(str(opath), out_img)
        if not ok:
            print(f"[warn] failed to write {opath}")
            continue

        # copy matching label (same stem, .txt) if exists
        l_in = in_labs / rel.with_suffix(".txt")
        if l_in.exists():
            l_out = out_labs / rel.with_suffix(".txt")
            l_out.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(l_in, l_out)

        written += 1
        if i % 200 == 0 or i == len(img_paths):
            print(f"[{i}/{len(img_paths)}] wrote {opath}")

    print(f"\n✅ Noisy YOLO split created at: {out_root}")
    print(f"   Images written: {written}")
    print(f"   Labels copied : {(out_labs).rglob('*.txt').__length_hint__() if hasattr((out_labs).rglob('*.txt'), '__length_hint__') else 'OK'}")


def main():
    ap = argparse.ArgumentParser(description="Create a noisy/corrupted copy of a YOLO split.")
    ap.add_argument("--in-root", required=True, help="Path to split with images/ and labels/")
    ap.add_argument("--out-root", required=True, help="Output split (will be created)")
    ap.add_argument("--severity", type=int, default=2, help="Distortion severity 1..5 (default=2)")
    ap.add_argument("--seed", type=int, default=42, help="Random seed")
    args = ap.parse_args()

    add_noise_yolo(args.in_root, args.out_root, args.severity, args.seed)


if __name__ == "__main__":
    main()
