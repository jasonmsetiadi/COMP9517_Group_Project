20-25% positive, 75-80% negative samples for each training set
positive samples are just the ground truth bounding boxes + proposals with iou > 0.8
if no positive samples, use hard negative mining (use ground truth box)
if positive examples exist, negative samples are proposals wih iou < 0.3 (num_samples = min(int(num_pos*0.1), 1))

use LBP+SIFT+COLOR features (250)