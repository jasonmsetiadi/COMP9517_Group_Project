"""
Detection/Inference script
"""
import torch
import cv2
import numpy as np
from torchvision.ops import nms

import config
from model import create_model, load_checkpoint


def preprocess_image(image_path, image_size):
    """Load and preprocess image"""
    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    orig_h, orig_w = image_rgb.shape[:2]
    
    # Resize
    resized = cv2.resize(image_rgb, (image_size, image_size))
    
    # Normalize
    normalized = resized.astype(np.float32) / 255.0
    tensor = torch.from_numpy(normalized).permute(2, 0, 1)
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    tensor = (tensor - mean) / std
    
    return tensor.unsqueeze(0), image, (orig_h, orig_w)


def postprocess_detections(detections, orig_size, image_size, conf_thresh, nms_thresh):
    """Process model outputs"""
    if len(detections) == 0 or len(detections[0]) == 0:
        return [], [], []
    
    det = detections[0]
    boxes = det[:, :4]
    scores = det[:, 4]
    labels = det[:, 5].long()
    
    # Filter by confidence
    keep = scores > conf_thresh
    boxes, scores, labels = boxes[keep], scores[keep], labels[keep]
    
    if len(boxes) == 0:
        return [], [], []
    
    # Apply NMS
    keep_nms = nms(boxes, scores, nms_thresh)
    boxes, scores, labels = boxes[keep_nms], scores[keep_nms], labels[keep_nms]
    
    # Scale to original size
    orig_h, orig_w = orig_size
    boxes[:, [0, 2]] *= (orig_w / image_size)
    boxes[:, [1, 3]] *= (orig_h / image_size)
    
    return boxes.cpu().numpy(), scores.cpu().numpy(), labels.cpu().numpy()


def draw_detections(image, boxes, scores, labels, class_names):
    """Draw bounding boxes on image"""
    for box, score, label in zip(boxes, scores, labels):
        x1, y1, x2, y2 = map(int, box)
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        text = f"{class_names[int(label)]}: {score:.2f}"
        cv2.putText(image, text, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 
                   0.5, (0, 255, 0), 2)
    return image


def detect_image(checkpoint_path, image_path, output_path=None):
    """Run detection on single image"""
    device = torch.device(config.DEVICE)
    
    # Load model
    model = create_model(pretrained=False).to(device)
    model = load_checkpoint(model, checkpoint_path)
    model.eval()
    
    # Preprocess
    image_tensor, orig_image, orig_size = preprocess_image(image_path, config.IMAGE_SIZE)
    image_tensor = image_tensor.to(device)
    
    # Detect
    with torch.no_grad():
        detections = model.model(image_tensor)
    
    # Postprocess
    boxes, scores, labels = postprocess_detections(
        detections, orig_size, config.IMAGE_SIZE, 
        config.CONF_THRESHOLD, config.NMS_THRESHOLD
    )
    
    print(f"Detected {len(boxes)} objects")
    for i, (box, score, label) in enumerate(zip(boxes, scores, labels)):
        print(f"  {i+1}. {config.CLASS_NAMES[int(label)]}: {score:.3f}")
    
    # Draw and save
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