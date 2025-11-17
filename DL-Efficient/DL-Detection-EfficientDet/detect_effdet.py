"""
Detection/Inference script (FINAL FIX)
"""
import torch
import cv2
import numpy as np
from torchvision.ops import nms
from effdet import DetBenchPredict 

import config
from model import create_model, load_checkpoint


def preprocess_image(image_path, image_size):
    """Load and preprocess image"""
    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    orig_h, orig_w = image_rgb.shape[:2]
    
    resized = cv2.resize(image_rgb, (image_size, image_size))
    
    normalized = resized.astype(np.float32) / 255.0
    tensor = torch.from_numpy(normalized).permute(2, 0, 1)
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    tensor = (tensor - mean) / std
    
    return tensor.unsqueeze(0), image, (orig_h, orig_w)


def postprocess_detections(detections, orig_size, image_size, conf_thresh, nms_thresh):
    """Process model outputs from DetBenchPredict"""
    
    if not torch.is_tensor(detections):
         return np.array([]), np.array([]), np.array([])
         
    detections = detections[0] 
    
    if len(detections) == 0:
        return np.array([]), np.array([]), np.array([])
    
    boxes = detections[:, :4]
    scores = detections[:, 4]
    labels = detections[:, 5].long()
    
    keep = scores > conf_thresh
    boxes, scores, labels = boxes[keep], scores[keep], labels[keep]
    
    if len(boxes) == 0:
        return np.array([]), np.array([]), np.array([])
    
    orig_h, orig_w = orig_size
    
    if isinstance(image_size, int):
        img_h = img_w = image_size
    else:
        img_h, img_w = image_size

    boxes[:, [0, 2]] *= (orig_w / img_w)
    boxes[:, [1, 3]] *= (orig_h / img_h)
    
    return boxes.cpu().numpy(), scores.cpu().numpy(), labels.cpu().numpy()


def draw_detections(image, boxes, scores, labels, class_names):
    """Draw bounding boxes on image"""
    for box, score, label in zip(boxes, scores, labels):
        x1, y1, x2, y2 = map(int, box)
        
        # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        # THIS IS THE FIX
        # Convert 1-indexed (1-12) to 0-indexed (0-11)
        # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        fixed_label = int(label) - 1
        
        if 0 <= fixed_label < len(class_names):
            class_name = class_names[fixed_label]
            text = f"{class_name}: {score:.2f}"
            
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(image, text, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 
                        0.5, (0, 255, 0), 2)
    return image


def detect_image(checkpoint_path, image_path, output_path=None):
    """Run detection on single image"""
    device = torch.device(config.DEVICE)
    
    train_model = create_model(pretrained=False).to(device)
    train_model = load_checkpoint(train_model, checkpoint_path)
    train_model.eval()

    model = DetBenchPredict(train_model.model).to(device)
    model.eval()
    
    image_tensor, orig_image, orig_size = preprocess_image(image_path, config.IMAGE_SIZE)
    image_tensor = image_tensor.to(device)
    
    with torch.no_grad():
        detections = model(image_tensor) 
    
    boxes, scores, labels = postprocess_detections(
        detections, orig_size, config.IMAGE_SIZE, 
        config.CONF_THRESHOLD, config.NMS_THRESHOLD
    )
    
    print(f"Detected {len(boxes)} objects")
    for i, (box, score, label) in enumerate(zip(boxes, scores, labels)):
        
        # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        # THIS IS THE FIX
        # Convert 1-indexed (1-12) to 0-indexed (0-11)
        # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        fixed_label = int(label) - 1
        
        if 0 <= fixed_label < len(config.CLASS_NAMES):
            class_name = config.CLASS_NAMES[fixed_label]
            print(f"  {i+1}. {class_name}: {score:.3f}")
    
    result_image = draw_detections(orig_image, boxes, scores, labels, config.CLASS_NAMES)
    result_image = cv2.cvtColor(result_image, cv2.COLOR_RGB2BGR)
    
    if output_path:
        cv2.imwrite(output_path, result_image)
        print(f"Saved to: {output_path}")
    
    return boxes, scores, labels


if __name__ == '__main__':
    import sys
    
    if len(sys.argv) < 3:
        print("Usage: python detect_effdet.py <checkpoint> <image_path> [output_path]")
        sys.exit(1)
    
    checkpoint = sys.argv[1]
    image_path = sys.argv[2]
    output_path = sys.argv[3] if len(sys.argv) > 3 else 'detection_result.jpg'
    
    detect_image(checkpoint, image_path, output_path)