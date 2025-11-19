10% positive, 90% negative samples for each training set
positive samples are just the ground truth bounding boxes
negative samples are proposals wih iou < 0.1
num_samples = 2 if num_pos == 0 else num_pos*2
