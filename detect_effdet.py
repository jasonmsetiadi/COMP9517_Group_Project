import torch
import cv2
import numpy as np
from effdet import create_model
import albumentations as A
from albumentations.pytorch import ToTensorV2 # Better for this

def detect_image(model_path, img_path, num_classes=12, img_size=640, score_threshold=0.3):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 1. --- Define the EXACT same transforms as in training ---
    #    (but without augmentations like flips, etc.)
    transform = A.Compose([
        A.LongestMaxSize(max_size=img_size),
        A.PadIfNeeded(img_size, img_size, border_mode=cv2.BORDER_CONSTANT, value=(0, 0, 0)),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)), # Use ImageNet stats
        ToTensorV2()
    ])

    model = create_model(
        'tf_efficientdet_d0',
        bench_task='predict',
        num_classes=num_classes,
        pretrained=False,
        image_size=(img_size, img_size)
    ).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # 2. --- Load and transform the image ---
    img_bgr = cv2.imread(img_path)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    orig_h, orig_w = img_rgb.shape[:2]

    # Apply transforms
    transformed = transform(image=img_rgb)
    x = transformed['image'].unsqueeze(0).to(device)
    
    # 3. --- Run prediction ---
    with torch.no_grad():
        # Detections are [N, 6] -> [x1, y1, x2, y2, score, class]
        detections = model(x)[0] # Get detections for the first (only) image
    
    # 4. --- Process and draw detections ---
    detections = detections.cpu().numpy()
    
    # Calculate scale factor to map boxes back to original image size
    # We used LongestMaxSize, so the scale is based on the longest side
    scale = min(img_size / orig_w, img_size / orig_h)

    for det in detections:
        score = det[4]
        if score < score_threshold:
            continue
            
        box = det[0:4]
        class_id = int(det[5])
        
        # Scale boxes back to original image coordinates
        # Note: This doesn't account for padding. A more robust way
        # is to also pass back the 'scale' and 'pad' from the dataset.
        # But for visualization, scaling from the resized (640, 640)
        # to the original (orig_w, orig_h) is simpler.
        
        # Simpler scaling (assumes box coords are relative to 640x640):
        # We need to account for the padding.
        # The 'albumentations' transform resizes and then pads.
        # Let's find the new size *after* resizing but *before* padding.
        
        new_w = int(orig_w * scale)
        new_h = int(orig_h * scale)
        
        # Calculate padding
        pad_x = (img_size - new_w) // 2
        pad_y = (img_size - new_h) // 2
        
        # 1. Remove padding
        box[0] = box[0] - pad_x
        box[1] = box[1] - pad_y
        box[2] = box[2] - pad_x
        box[3] = box[3] - pad_y
        
        # 2. Scale back to original size
        box[0] = box[0] / scale
        box[1] = box[1] / scale
        box[2] = box[2] / scale
        box[3] = box[3] / scale

        # Clip to image bounds
        box = np.clip(box, [0, 0, 0, 0], [orig_w, orig_h, orig_w, orig_h]).astype(int)

        # Draw on the *original* BGR image
        cv2.rectangle(img_bgr, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)
        cv2.putText(img_bgr, f"Class {class_id} ({score:.2f})", (box[0], box[1] - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    print(f"Found {len(detections[detections[:, 4] >= score_threshold])} objects.")
    cv2.imwrite("detection_result.png", img_bgr)
    print("✅ Detection complete! Result saved as detection_result.png")

if __name__ == "__main__":
    model_file = "efficientdet_best.pth"  # <-- Use your newly trained model
    image_to_test = "C:/path/to/AgroPest-12/test/images/some_test_image.png" # <-- Put the path here

    detect_image(model_file, image_to_test, num_classes=12)
