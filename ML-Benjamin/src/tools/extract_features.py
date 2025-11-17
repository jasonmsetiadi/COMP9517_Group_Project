import cv2
import numpy as np
from skimage.feature import hog

def extract_hog_features(image_list):
    features = []
    for img in image_list:

        # If image is grayscale (2D), skip cvtColor
        if len(img.shape) == 2:
            gray = img
        else:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        gray = cv2.resize(gray, (128, 128))

        hog_feat = hog(
            gray,
            pixels_per_cell=(16, 16),
            cells_per_block=(2, 2),
            orientations=9,
            visualize=False
        )
        features.append(hog_feat)

    return np.array(features)


def extract_color_hist_features(image_list, bins=16):
    """
    Extract flattened HSV color histograms for a list of images.
    """
    features = []
    for img in image_list:
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        hist = []
        for i in range(3):
            h = cv2.calcHist([hsv], [i], None, [bins], [0,256])
            h = cv2.normalize(h, h).flatten()
            hist.extend(h)
        features.append(np.array(hist))
    return np.array(features)

def extract_combined_features(image_list):
    """
    Combine HOG + color histograms for each image.
    """
    hog_feats = extract_hog_features(image_list)
    color_feats = extract_color_hist_features(image_list)
    combined = np.hstack([hog_feats, color_feats])
    return combined
