#!/usr/bin/env python3
"""
CSc 8830 (Computer Vision) — Module 4 Assignment
Classical (non-ML) thermal animal boundary extraction using OpenCV.

How to run (single image):
    python thermal_animal_segmentation.py --input data/sample_inputs/01_dog_crop.png --output_dir outputs/classical --show 0

How to run (folder):
    python thermal_animal_segmentation.py --input data/sample_inputs --output_dir outputs/classical

Key idea:
Thermal pseudo-color images (Spectra palette) typically exhibit a *bimodal-ish* intensity distribution:
cool background is darker, warm bodies are brighter. Otsu thresholding often separates these modes well.
We then clean the binary mask with morphology and extract precise object contours.

IMPORTANT (assignment constraint):
- This script uses ONLY OpenCV + NumPy (classical CV). No deep learning / ML models.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np


# -----------------------------
# Data structures
# -----------------------------
@dataclass
class PipelineParams:
    blur_ksize: int = 7
    blur_sigma: float = 0.0
    threshold: str = "otsu"  # "otsu" | "adaptive"
    adaptive_block: int = 31
    adaptive_C: int = 2

    morph_kernel: int = 7
    open_iters: int = 1
    close_iters: int = 2
    dilate_iters: int = 1
    erode_iters: int = 0

    min_area: int = 500
    max_area: int = 10_000_000
    select: str = "largest"  # "largest" | "all"
    top_k: int = 5  # for select="all": keep up to top_k by area (after filtering)

    mask_alpha: float = 0.45
    contour_thickness: int = 2


# -----------------------------
# IO helpers
# -----------------------------
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def list_images(input_path: Path) -> List[Path]:
    if input_path.is_file():
        return [input_path]
    if not input_path.is_dir():
        raise FileNotFoundError(f"Input path not found: {input_path}")
    files = []
    for p in sorted(input_path.rglob("*")):
        if p.suffix.lower() in IMG_EXTS:
            files.append(p)
    return files


def safe_imread(path: Path) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"OpenCV failed to read image: {path}")
    return img


def ensure_odd(x: int) -> int:
    return x if x % 2 == 1 else x + 1


def mkdir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


# -----------------------------
# Core pipeline
# -----------------------------
def to_grayscale(bgr: np.ndarray) -> np.ndarray:
    # Thermal images are often false-colored; grayscale collapses the palette to intensity,
    # which (for Spectra-like palettes) typically correlates with "warmth".
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)


def gaussian_blur(gray: np.ndarray, ksize: int, sigma: float) -> np.ndarray:
    ksize = ensure_odd(max(3, int(ksize)))
    return cv2.GaussianBlur(gray, (ksize, ksize), sigmaX=sigma, sigmaY=sigma)


def otsu_threshold(blurred: np.ndarray) -> Tuple[np.ndarray, float]:
    # Otsu chooses threshold t that minimizes within-class variance (equiv. maximizes between-class variance)
    # for a 2-class partition of the histogram.
    t, bin_mask = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Heuristic polarity fix:
    # If foreground occupies "too much" area, likely threshold picked background as white -> invert.
    fg_ratio = float(np.count_nonzero(bin_mask)) / float(bin_mask.size)
    if fg_ratio > 0.55:
        bin_mask = cv2.bitwise_not(bin_mask)
    return bin_mask, float(t)


def adaptive_threshold(blurred: np.ndarray, block: int, C: int) -> Tuple[np.ndarray, float]:
    block = ensure_odd(max(3, int(block)))
    bin_mask = cv2.adaptiveThreshold(
        blurred,
        255,
        adaptiveMethod=cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        thresholdType=cv2.THRESH_BINARY,
        blockSize=block,
        C=int(C),
    )
    fg_ratio = float(np.count_nonzero(bin_mask)) / float(bin_mask.size)
    if fg_ratio > 0.55:
        bin_mask = cv2.bitwise_not(bin_mask)
    return bin_mask, float("nan")


def morphology_cleanup(bin_mask: np.ndarray, k: int, open_iters: int, close_iters: int,
                       dilate_iters: int, erode_iters: int) -> np.ndarray:
    k = max(3, int(k))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))

    out = bin_mask.copy()

    # Opening removes small bright speckles (erosion then dilation)
    if open_iters > 0:
        out = cv2.morphologyEx(out, cv2.MORPH_OPEN, kernel, iterations=int(open_iters))

    # Closing fills small holes inside the warm-object blob (dilation then erosion)
    if close_iters > 0:
        out = cv2.morphologyEx(out, cv2.MORPH_CLOSE, kernel, iterations=int(close_iters))

    # Mild dilation connects fragmented warm regions (e.g., legs/ears) into a single component
    if dilate_iters > 0:
        out = cv2.dilate(out, kernel, iterations=int(dilate_iters))

    # Optional erosion to undo over-dilation if needed
    if erode_iters > 0:
        out = cv2.erode(out, kernel, iterations=int(erode_iters))

    return out


def find_contours(bin_mask: np.ndarray) -> List[np.ndarray]:
    # External contours are usually sufficient for object boundary extraction.
    contours, _ = cv2.findContours(bin_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return list(contours)


def contour_stats(contour: np.ndarray) -> Dict[str, float]:
    area = float(cv2.contourArea(contour))
    if area <= 0:
        return {"area": 0.0, "solidity": 0.0, "circularity": 0.0, "aspect": 0.0}
    x, y, w, h = cv2.boundingRect(contour)
    hull = cv2.convexHull(contour)
    hull_area = float(cv2.contourArea(hull)) if hull is not None else area
    solidity = area / hull_area if hull_area > 1e-6 else 0.0
    perimeter = float(cv2.arcLength(contour, True))
    circularity = 4.0 * np.pi * area / (perimeter * perimeter + 1e-6)
    aspect = float(w) / float(h + 1e-6)
    return {"area": area, "solidity": solidity, "circularity": circularity, "aspect": aspect}


def filter_contours(
    contours: Sequence[np.ndarray],
    min_area: int,
    max_area: int,
) -> List[np.ndarray]:
    kept = []
    for c in contours:
        stats = contour_stats(c)
        area = stats["area"]
        if area < float(min_area) or area > float(max_area):
            continue

        # Light geometric sanity checks help suppress obvious background blobs.
        # Thermal animals tend to be moderately compact (not long thin lines).
        if stats["solidity"] < 0.25:
            continue
        if stats["circularity"] < 0.02:
            continue

        kept.append(c)
    return kept


def select_contours(contours: List[np.ndarray], select: str, top_k: int) -> List[np.ndarray]:
    if not contours:
        return []
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    if select == "largest":
        return [contours[0]]
    return contours[: max(1, int(top_k))]


def contours_to_mask(shape_hw: Tuple[int, int], contours: Sequence[np.ndarray]) -> np.ndarray:
    h, w = shape_hw
    mask = np.zeros((h, w), dtype=np.uint8)
    if contours:
        cv2.drawContours(mask, list(contours), contourIdx=-1, color=255, thickness=-1)
    return mask


def overlay_mask(bgr: np.ndarray, mask: np.ndarray, alpha: float) -> np.ndarray:
    alpha = float(np.clip(alpha, 0.0, 1.0))
    overlay = bgr.copy()
    color = np.array([0, 255, 0], dtype=np.uint8)  # green overlay for mask region

    # Apply alpha blending only where mask is 1
    m = mask.astype(bool)
    overlay[m] = (alpha * color + (1.0 - alpha) * overlay[m]).astype(np.uint8)
    return overlay


def draw_contours(bgr: np.ndarray, contours: Sequence[np.ndarray], thickness: int) -> np.ndarray:
    out = bgr.copy()
    if contours:
        cv2.drawContours(out, list(contours), -1, (0, 0, 255), int(thickness))  # red boundary
    return out


def bgr_from_gray(gray: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)


def make_side_by_side(panels: List[Tuple[str, np.ndarray]], max_width: int = 1800) -> np.ndarray:
    """
    Create a labeled horizontal strip (resizes panels to common height).
    """
    resized = []
    target_h = min(500, max(p.shape[0] for _, p in panels))
    font = cv2.FONT_HERSHEY_SIMPLEX

    for label, img in panels:
        if img.ndim == 2:
            img = bgr_from_gray(img)
        scale = target_h / img.shape[0]
        new_w = max(1, int(img.shape[1] * scale))
        r = cv2.resize(img, (new_w, target_h), interpolation=cv2.INTER_AREA)
        cv2.putText(r, label, (10, 30), font, 1.0, (255, 255, 255), 2, cv2.LINE_AA)
        resized.append(r)

    strip = cv2.hconcat(resized)

    if strip.shape[1] > max_width:
        scale = max_width / strip.shape[1]
        strip = cv2.resize(strip, (max_width, int(strip.shape[0] * scale)), interpolation=cv2.INTER_AREA)
    return strip


def run_pipeline(bgr: np.ndarray, params: PipelineParams) -> Dict[str, np.ndarray]:
    gray = to_grayscale(bgr)
    blurred = gaussian_blur(gray, params.blur_ksize, params.blur_sigma)

    if params.threshold == "adaptive":
        thresh, tval = adaptive_threshold(blurred, params.adaptive_block, params.adaptive_C)
    else:
        thresh, tval = otsu_threshold(blurred)

    morph = morphology_cleanup(
        thresh,
        k=params.morph_kernel,
        open_iters=params.open_iters,
        close_iters=params.close_iters,
        dilate_iters=params.dilate_iters,
        erode_iters=params.erode_iters,
    )

    contours = find_contours(morph)
    contours = filter_contours(contours, params.min_area, params.max_area)
    contours = select_contours(contours, params.select, params.top_k)

    mask = contours_to_mask((bgr.shape[0], bgr.shape[1]), contours)
    overlay = overlay_mask(bgr, mask, params.mask_alpha)
    contour_vis = draw_contours(overlay, contours, params.contour_thickness)

    side_by_side = make_side_by_side(
        [
            ("Original", bgr),
            ("Grayscale", gray),
            ("Otsu/Adaptive", thresh),
            ("Morphology", morph),
            ("Final Overlay", contour_vis),
        ]
    )

    return {
        "original": bgr,
        "gray": gray,
        "blurred": blurred,
        "threshold": thresh,
        "morph": morph,
        "mask": mask,
        "overlay": contour_vis,
        "side_by_side": side_by_side,
        "otsu_threshold": np.array([tval], dtype=np.float32),
    }


def save_outputs(out_dir: Path, stem: str, outputs: Dict[str, np.ndarray]) -> Dict[str, str]:
    mkdir(out_dir)

    paths = {}
    def _save(key: str, filename: str):
        p = out_dir / filename
        img = outputs[key]
        if key == "otsu_threshold":
            return
        cv2.imwrite(str(p), img)
        paths[key] = str(p)

    _save("original", f"{stem}_01_original.png")
    _save("gray", f"{stem}_02_gray.png")
    _save("threshold", f"{stem}_03_threshold.png")
    _save("morph", f"{stem}_04_morph.png")
    _save("mask", f"{stem}_05_mask.png")
    _save("overlay", f"{stem}_06_overlay.png")
    _save("side_by_side", f"{stem}_07_side_by_side.png")
    return paths


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="OpenCV-only thermal animal boundary extraction (Otsu + morphology + contours)."
    )
    p.add_argument("--input", required=True, help="Input image path or folder.")
    p.add_argument("--output_dir", default="outputs/classical", help="Output directory.")
    p.add_argument("--show", type=int, default=0, help="Show OpenCV windows (1=yes, 0=no).")

    # Pipeline parameters
    p.add_argument("--threshold", choices=["otsu", "adaptive"], default="otsu")
    p.add_argument("--blur_ksize", type=int, default=7)
    p.add_argument("--blur_sigma", type=float, default=0.0)

    p.add_argument("--adaptive_block", type=int, default=31)
    p.add_argument("--adaptive_C", type=int, default=2)

    p.add_argument("--morph_kernel", type=int, default=7)
    p.add_argument("--open_iters", type=int, default=1)
    p.add_argument("--close_iters", type=int, default=2)
    p.add_argument("--dilate_iters", type=int, default=1)
    p.add_argument("--erode_iters", type=int, default=0)

    p.add_argument("--min_area", type=int, default=500, help="Min contour area to keep.")
    p.add_argument("--max_area", type=int, default=10_000_000, help="Max contour area to keep.")
    p.add_argument("--select", choices=["largest", "all"], default="largest")
    p.add_argument("--top_k", type=int, default=5, help="If select=all, keep up to top_k.")

    p.add_argument("--mask_alpha", type=float, default=0.45)
    p.add_argument("--contour_thickness", type=int, default=2)

    return p.parse_args()


def main() -> None:
    args = parse_args()

    input_path = Path(args.input)
    out_root = Path(args.output_dir)
    mkdir(out_root)

    params = PipelineParams(
        blur_ksize=args.blur_ksize,
        blur_sigma=args.blur_sigma,
        threshold=args.threshold,
        adaptive_block=args.adaptive_block,
        adaptive_C=args.adaptive_C,
        morph_kernel=args.morph_kernel,
        open_iters=args.open_iters,
        close_iters=args.close_iters,
        dilate_iters=args.dilate_iters,
        erode_iters=args.erode_iters,
        min_area=args.min_area,
        max_area=args.max_area,
        select=args.select,
        top_k=args.top_k,
        mask_alpha=args.mask_alpha,
        contour_thickness=args.contour_thickness,
    )

    images = list_images(input_path)
    if not images:
        raise SystemExit(f"No images found under: {input_path}")

    run_summary = []
    for img_path in images:
        t0 = time.perf_counter()
        bgr = safe_imread(img_path)
        outputs = run_pipeline(bgr, params)
        dt_ms = (time.perf_counter() - t0) * 1000.0

        stem = img_path.stem.replace(" ", "_")
        out_dir = out_root / stem
        saved = save_outputs(out_dir, stem, outputs)

        mask_area = int(np.count_nonzero(outputs["mask"]))
        run_summary.append(
            {
                "image": str(img_path),
                "output_dir": str(out_dir),
                "ms": dt_ms,
                "mask_area_px": mask_area,
                "saved": saved,
            }
        )

        if args.show == 1:
            cv2.imshow("Thermal Boundary (side-by-side)", outputs["side_by_side"])
            key = cv2.waitKey(0)
            if key == 27:  # ESC
                break

    if args.show == 1:
        cv2.destroyAllWindows()

    # Write a small summary JSON (useful for reporting / timing).
    summary_path = out_root / "run_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(run_summary, f, indent=2)
    print(f"[OK] Processed {len(run_summary)} images. Summary: {summary_path}")


if __name__ == "__main__":
    main()
