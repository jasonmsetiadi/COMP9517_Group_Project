import cv2
import numpy as np
from skimage.feature import hog, local_binary_pattern
from typing import Tuple
import matplotlib.pyplot as plt

# function to get warped region given image and bounding box (IoU format)
def warp_region(
    image: np.ndarray,
    box: Tuple[int, int, int, int], # IoU format box
    output_size: Tuple[int, int] = None,
    context: int = 0
) -> np.ndarray:
    """Warp an image region (proposal) to a fixed size (e.g. 227x227), with context padding as used in R-CNN."""

    H, W = image.shape[:2]
    x_min, y_min, x_max, y_max = box
    x_min, y_min, x_max, y_max = map(int, [x_min, y_min, x_max, y_max])


    # Expand box by context pixels (p = 16 recommended)
    x0 = max(0, x_min - context)
    y0 = max(0, y_min - context)
    x1 = min(W, x_max + context)
    y1 = min(H, y_max + context)

    # Crop region with padding
    crop = image[y0:y1, x0:x1]

    # Resize (warp) to fixed CNN input size
    if output_size is not None:
        # If the crop is empty (shouldn’t happen, but just in case)
        if crop.size == 0:
            crop = np.zeros((output_size[1], output_size[0], 3), dtype=np.uint8)
        warped = cv2.resize(crop, output_size, interpolation=cv2.INTER_LINEAR)
    else:
        warped = crop

    return warped

def extract_hog_features(image):
    """Compute HOG (Histogram of Oriented Gradients) features."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    features = hog(
        gray,
        orientations=9,
        pixels_per_cell=(8, 8),
        cells_per_block=(2, 2),
        block_norm='L2-Hys',
        transform_sqrt=True,
        feature_vector=True
    )
    return features


def extract_color_histogram(image: np.ndarray, bins: int = 32) -> np.ndarray:
    """Extract color histogram features."""
    hist_features = []
    for i in range(3):  # RGB channels
        hist = cv2.calcHist([image], [i], None, [bins], [0, 256])
        hist = hist.flatten()
        hist = hist / (hist.sum() + 1e-7)  # Normalize
        hist_features.append(hist)
    
    return np.concatenate(hist_features)


def extract_sift_features(image):
    """Compute SIFT descriptors and flatten by averaging (fixed length)."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    sift = cv2.SIFT_create()
    keypoints, descriptors = sift.detectAndCompute(gray, None)
    if descriptors is None:
        # No keypoints detected → return zeros
        return np.zeros(128)
    # Average pool to fixed size
    desc_mean = np.mean(descriptors, axis=0)
    return desc_mean

def extract_lbp_features(image, num_points=24, radius=3):
    """Compute Local Binary Pattern histogram (texture features)."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    lbp = local_binary_pattern(gray, num_points, radius, method='uniform')
    (hist, _) = np.histogram(
        lbp.ravel(),
        bins=np.arange(0, num_points + 3),
        range=(0, num_points + 2)
    )
    hist = hist.astype("float")
    hist /= (hist.sum() + 1e-6)
    return hist

def extract_features(image: np.ndarray, use_hog: bool = True, use_sift: bool = False,use_lbp: bool = False, use_color: bool = False) -> np.ndarray:
        """
        Combine all features into a single vector.
        """
        features_list = []
        
        if use_hog:
            hog_features = extract_hog_features(image)
            features_list.append(hog_features)
        
        if use_sift:
            sift_features = extract_sift_features(image)
            features_list.append(sift_features)
        
        if use_lbp:
            lbp_features = extract_lbp_features(image)
            features_list.append(lbp_features)
        
        if use_color:
            color_features = extract_color_histogram(image, bins=32)
            features_list.append(color_features)
        
        if len(features_list) == 0:
            raise ValueError("At least one feature type must be enabled")
            
        # Combine all features
        combined_features = np.concatenate(features_list)
        
        return combined_features

def visualize_bbox(img_path, boxes):
    # load image
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # for each bbox, warp the region
    for bbox in boxes:
        warped = warp_region(img, bbox, context=0)
        # visualize beside original image with bounding box
        fig, axs = plt.subplots(1, 2, figsize=(10, 5))
        axs[0].imshow(img)
        axs[0].add_patch(plt.Rectangle((bbox[0], bbox[1]), bbox[2] - bbox[0], bbox[3] - bbox[1], fill=False, color='red', linewidth=2))
        axs[1].imshow(warped)
        plt.show()