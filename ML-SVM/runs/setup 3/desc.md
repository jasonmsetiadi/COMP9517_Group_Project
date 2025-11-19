50% positive, 50% negative samples for each training set
positive samples are just the ground truth bounding boxes + proposals with iou > 0.7
negative samples are proposals wih iou < 0.3
num_samples = 1 if num_pos == 0 else min(int(num_pos*0.1), 1)