"""
rcnn_selective_search_visualize_scaled.py

Implements the first stage of R-CNN (Girshick et al., 2014):
 - Selective Search region proposal
 - Resize image to width 500 px (as in paper)
 - Map proposals back to original image size
 - Visualize bounding boxes on original image

Requirements:
  pip install opencv-contrib-python numpy
"""

import os
import cv2
import numpy as np
from typing import List, Tuple
import matplotlib.pyplot as plt
import matplotlib.cm as cm  # for color maps

BBox = Tuple[int, int, int, int]  # (x, y, w, h)

# ---------- Utility Functions ----------

def resize_keep_aspect(img: np.ndarray, target_width: int = 500) -> Tuple[np.ndarray, float]:
    """Resize image keeping aspect ratio. Return resized image and scale factor (orig→resized)."""
    h, w = img.shape[:2]
    if w == target_width:
        return img, 1.0
    scale = target_width / float(w)
    new_h = int(round(h * scale))
    resized = cv2.resize(img, (target_width, new_h))
    return resized, scale


def run_selective_search(img: np.ndarray, mode: str = "fast") -> List[BBox]:
    """Run OpenCV Selective Search on an image.
    Output boxes are in the format of (x, y, w, h) where x, y are the top-left corner of the box, w, h are the width and height of the box.
    """
    if not hasattr(cv2, "ximgproc"):
        raise RuntimeError("You need OpenCV contrib. Install with: pip install opencv-contrib-python")

    ss = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()
    ss.setBaseImage(img)
    if mode == "fast":
        ss.switchToSelectiveSearchFast()
    elif mode == "quality":
        ss.switchToSelectiveSearchQuality()
    else:
        raise ValueError("mode must be 'fast' or 'quality'")

    rects = ss.process()
    boxes = [(int(x), int(y), int(w), int(h)) for (x, y, w, h) in rects]
    return boxes


def filter_boxes(boxes: List[BBox], min_size: int = 20, max_boxes: int = 2000) -> List[BBox]:
    """Remove small boxes and keep up to max_boxes sorted by area."""
    filtered = [b for b in boxes if b[2] >= min_size and b[3] >= min_size]
    filtered.sort(key=lambda b: b[2] * b[3], reverse=True)
    return filtered[:max_boxes]


def map_boxes_to_original(boxes: List[BBox], scale: float) -> List[BBox]:
    """Map boxes from resized image to original coordinates."""
    inv_scale = 1.0 / scale
    mapped = []
    for (x, y, w, h) in boxes:
        mapped.append((
            int(round(x * inv_scale)),
            int(round(y * inv_scale)),
            int(round(w * inv_scale)),
            int(round(h * inv_scale))
        ))
    return mapped


def visualize_boxes(img: np.ndarray, boxes: List[BBox], num_to_show: int = None):
    """Draw bounding boxes on the image."""
    vis = img.copy()
    total = len(boxes) if num_to_show is None else min(num_to_show, len(boxes))
    cmap = cm.get_cmap('plasma', total)  # choose your favorite: 'viridis', 'plasma', 'turbo', etc.

    for i, (x, y, w, h) in enumerate(boxes[:total]):
        # Normalize index to [0, 1] for colormap
        color_float = cmap(i / total)[:3]  # RGB in [0,1]
        color_bgr = tuple(int(255 * c) for c in color_float[::-1])  # convert to BGR for OpenCV
        cv2.rectangle(vis, (x, y), (x + w, y + h), color_bgr, 2)

    # Add colorbar legend
    legend = np.zeros((50, vis.shape[1], 3), dtype=np.uint8)
    for x in range(legend.shape[1]):
        c = cmap(x / legend.shape[1])[:3]
        legend[:, x] = [int(255 * c[2]), int(255 * c[1]), int(255 * c[0])]
    vis = np.vstack([vis, legend])
    return vis


def warp_region(
    image: np.ndarray,
    box: Tuple[int, int, int, int],
    output_size: Tuple[int, int] = (227, 227),
    context: int = 16
) -> np.ndarray:
    """Warp an image region (proposal) to a fixed size (e.g. 227x227), with context padding as used in R-CNN."""

    H, W = image.shape[:2]
    x, y, w, h = box

    # Expand box by context pixels (p = 16 recommended)
    x0 = max(0, x - context)
    y0 = max(0, y - context)
    x1 = min(W, x + w + context)
    y1 = min(H, y + h + context)

    # Crop region with padding
    crop = image[y0:y1, x0:x1]

    # If the crop is empty (shouldn’t happen, but just in case)
    if crop.size == 0:
        crop = np.zeros((output_size[1], output_size[0], 3), dtype=np.uint8)

    # Resize (warp) to fixed CNN input size
    warped = cv2.resize(crop, output_size, interpolation=cv2.INTER_LINEAR)

    return warped


# ---------- Main Pipeline ----------

def selective_search_rcnn_pipeline(image_path: str,
                                   target_width: int = 500,
                                   mode: str = "fast",
                                   max_boxes: int = 2000,
                                   visualize: bool = False,
                                   save_visualization_path: str = None,
                                   save_boxes_path: str = None):
    """Run selective search, rescale boxes to original size, and visualize."""
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    if img is None:
        raise ValueError(f"Cannot read image: {image_path}")

    print("[INFO] Running Selective Search ...")
    resized, scale = resize_keep_aspect(img, target_width)
    boxes_resized = run_selective_search(resized, mode=mode)
    boxes_resized = filter_boxes(boxes_resized, min_size=20, max_boxes=max_boxes)
    print(f"[INFO] Got {len(boxes_resized)} region proposals on resized image.")

    # Map back to original coordinates
    boxes_original = map_boxes_to_original(boxes_resized, scale)
    print("[INFO] Boxes mapped back to original image coordinates.")

    # Visualize on original image
    vis = visualize_boxes(img, boxes_original)
    if visualize:
        plt.imshow(vis)
        plt.show()
    if save_visualization_path:
        cv2.imwrite(save_visualization_path, vis)
        print(f"[INFO] Saved visualization as {save_visualization_path}")
    if save_boxes_path: # given .txt file path, save the boxes to the file
        os.makedirs(os.path.dirname(save_boxes_path), exist_ok=True)
        with open(save_boxes_path, "w") as f:
            for box in boxes_original:
                f.write(f"{box[0]} {box[1]} {box[2]} {box[3]}\n")
        print(f"[INFO] Saved boxes as {save_boxes_path}")
    return boxes_original
