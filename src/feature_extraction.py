import cv2
import numpy as np
from skimage.feature import hog, local_binary_pattern

def extract_hog_features(image, resize_shape=(128, 128)):
    """Compute HOG (Histogram of Oriented Gradients) features."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, resize_shape)
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

def extract_sift_features(image, max_features=100):
    """Compute SIFT descriptors and flatten by averaging (fixed length)."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    sift = cv2.SIFT_create()
    keypoints, descriptors = sift.detectAndCompute(gray, None)
    if descriptors is None:
        # No keypoints detected → return zeros
        return np.zeros(128)
    # Take up to max_features descriptors
    descriptors = descriptors[:max_features]
    # Average pool to fixed size
    desc_mean = np.mean(descriptors, axis=0)
    return desc_mean

def extract_lbp_features(image, resize_shape=(128, 128), num_points=24, radius=3):
    """Compute Local Binary Pattern histogram (texture features)."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, resize_shape)
    lbp = local_binary_pattern(gray, num_points, radius, method='uniform')
    (hist, _) = np.histogram(
        lbp.ravel(),
        bins=np.arange(0, num_points + 3),
        range=(0, num_points + 2)
    )
    hist = hist.astype("float")
    hist /= (hist.sum() + 1e-6)
    return hist

def extract_combined_features(image):
    """Combine HOG + SIFT + LBP features into a single vector."""
    hog_f = extract_hog_features(image)
    sift_f = extract_sift_features(image)
    lbp_f = extract_lbp_features(image)
    # Concatenate all features
    combined = np.hstack([hog_f, sift_f, lbp_f])
    return combined
