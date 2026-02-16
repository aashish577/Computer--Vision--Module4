

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np

from thermal_animal_segmentation import (
    PipelineParams,
    list_images,
    mkdir,
    run_pipeline,
    safe_imread,
    overlay_mask,
    draw_contours,
    find_contours,
    filter_contours,
    select_contours,
    contours_to_mask,
    make_side_by_side,
)


def mask_to_box(mask: np.ndarray) -> Tuple[int, int, int, int]:
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return (0, 0, mask.shape[1] - 1, mask.shape[0] - 1)
    x0, x1 = int(xs.min()), int(xs.max())
    y0, y1 = int(ys.min()), int(ys.max())
    return (x0, y0, x1, y1)


def sample_points_from_mask(mask: np.ndarray, n_pos: int = 5, n_neg: int = 5) -> Tuple[np.ndarray, np.ndarray]:
    """
    Sample a few positive points inside the mask and negative points outside.
    Helps SAM2 when the box includes other warm objects (e.g., nearby human).
    """
    h, w = mask.shape
    rng = np.random.default_rng(0)

    pos = np.column_stack(np.where(mask > 0))
    neg = np.column_stack(np.where(mask == 0))

    def pick(arr, n):
        if len(arr) == 0:
            return np.zeros((0, 2), dtype=np.float32)
        idx = rng.choice(len(arr), size=min(n, len(arr)), replace=False)
        pts = arr[idx]
        # convert (y,x) -> (x,y)
        return np.stack([pts[:, 1], pts[:, 0]], axis=1).astype(np.float32)

    pos_pts = pick(pos, n_pos)
    neg_pts = pick(neg, n_neg)
    pts = np.vstack([pos_pts, neg_pts]) if (len(pos_pts) + len(neg_pts)) > 0 else np.zeros((0, 2), dtype=np.float32)
    labels = np.hstack([np.ones(len(pos_pts)), np.zeros(len(neg_pts))]).astype(np.int32)
    return pts, labels


def run_sam2_on_image(bgr: np.ndarray, box_xyxy: Tuple[int, int, int, int], points: np.ndarray, labels: np.ndarray,
                      sam2_model: str, device: str) -> np.ndarray:
    """
    Returns binary mask (uint8 0/255).
    """
    try:
        import torch
        from sam2.sam2_image_predictor import SAM2ImagePredictor
    except Exception as e:
        raise SystemExit(
            "SAM2 dependencies not found. Install requirements-sam2.txt and ensure torch works.\n"
            f"Original import error: {e}"
        )

    predictor = SAM2ImagePredictor.from_pretrained(sam2_model, device=device)

    # SAM2 expects RGB
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    # Some installations accept set_image(np.ndarray) directly.
    predictor.set_image(rgb)

    x0, y0, x1, y1 = box_xyxy
    box = np.array([x0, y0, x1, y1], dtype=np.float32)

    use_cuda = (device.startswith("cuda") and torch.cuda.is_available())
    autocast_ctx = torch.autocast("cuda", dtype=torch.bfloat16) if use_cuda else torch.autocast("cpu", enabled=False)

    with torch.inference_mode(), autocast_ctx:
        # Interface mirrors classic SAM:
        # predictor.predict(point_coords=..., point_labels=..., box=..., multimask_output=...)
        masks, scores, _ = predictor.predict(
            point_coords=points if len(points) > 0 else None,
            point_labels=labels if len(labels) > 0 else None,
            box=box[None, :],  # (1,4)
            multimask_output=False,
        )

    # masks shape is usually (N,H,W) boolean
    m = masks[0]
    if m.dtype != np.uint8:
        m = (m > 0).astype(np.uint8) * 255
    return m


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="SAM2 comparison for thermal animal segmentation.")
    p.add_argument("--input", required=True, help="Input image path or folder (same as classical).")
    p.add_argument("--output_dir", default="outputs/sam2", help="Output directory.")
    p.add_argument("--sam2_model", default="facebook/sam2-hiera-large", help="HuggingFace model id.")
    p.add_argument("--device", default="cuda", help="cuda or cpu")
    p.add_argument("--use_points", type=int, default=1, help="Use point prompts (1=yes, 0=no).")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    out_root = Path(args.output_dir)
    mkdir(out_root)

    # Use the same classical defaults.
    params = PipelineParams()

    images = list_images(input_path)
    if not images:
        raise SystemExit(f"No images found under: {input_path}")

    summary = []
    for img_path in images:
        bgr = safe_imread(img_path)

        # ---- Classical pipeline (for prompt + baseline comparison)
        t0 = time.perf_counter()
        classical = run_pipeline(bgr, params)
        classical_ms = (time.perf_counter() - t0) * 1000.0

        c_mask = classical["mask"]
        box = mask_to_box(c_mask)

        points, labels = (np.zeros((0, 2), dtype=np.float32), np.zeros((0,), dtype=np.int32))
        if args.use_points == 1:
            points, labels = sample_points_from_mask(c_mask, n_pos=5, n_neg=5)

        # ---- SAM2
        t1 = time.perf_counter()
        s_mask = run_sam2_on_image(bgr, box, points, labels, args.sam2_model, args.device)
        sam2_ms = (time.perf_counter() - t1) * 1000.0

        # ---- Visualizations
        s_overlay = overlay_mask(bgr, s_mask, alpha=0.45)
        s_contours = select_contours(filter_contours(find_contours(s_mask), 200, 10_000_000), "all", 25)
        s_vis = draw_contours(s_overlay, s_contours, thickness=2)

        # compute IoU between classical and SAM2 masks (purely for comparison)
        inter = np.logical_and(c_mask > 0, s_mask > 0).sum()
        union = np.logical_or(c_mask > 0, s_mask > 0).sum()
        iou = float(inter) / float(union + 1e-9)

        side = make_side_by_side(
            [
                ("Original", bgr),
                ("Classical (OpenCV)", classical["overlay"]),
                ("SAM2", s_vis),
            ],
            max_width=1800,
        )

        stem = img_path.stem.replace(" ", "_")
        out_dir = out_root / stem
        mkdir(out_dir)

        cv2.imwrite(str(out_dir / f"{stem}_01_original.png"), bgr)
        cv2.imwrite(str(out_dir / f"{stem}_02_classical_overlay.png"), classical["overlay"])
        cv2.imwrite(str(out_dir / f"{stem}_03_sam2_overlay.png"), s_vis)
        cv2.imwrite(str(out_dir / f"{stem}_04_sam2_mask.png"), s_mask)
        cv2.imwrite(str(out_dir / f"{stem}_05_side_by_side.png"), side)

        summary.append(
            {
                "image": str(img_path),
                "output_dir": str(out_dir),
                "classical_ms": classical_ms,
                "sam2_ms": sam2_ms,
                "iou_classical_vs_sam2": iou,
                "box_xyxy": box,
                "sam2_model": args.sam2_model,
            }
        )

        print(f"[OK] {img_path.name}: classical {classical_ms:.1f} ms | sam2 {sam2_ms:.1f} ms | IoU {iou:.3f}")

    with open(out_root / "run_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"[DONE] Wrote SAM2 comparison results to: {out_root}")


if __name__ == "__main__":
    main()
