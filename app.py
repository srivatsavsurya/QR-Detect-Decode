import torch
import torch.nn as nn
import torchvision.models as models
import cv2
import numpy as np
from torchvision import transforms
from pyzbar.pyzbar import decode as qr_decode

# Define the model architecture
class QRCodeDetector(nn.Module):
    def __init__(self):
        super(QRCodeDetector, self).__init__()
        self.model = models.detection.fasterrcnn_resnet50_fpn(pretrained=False)
    
    def forward(self, x):
        return self.model(x)
    

import torch

# Assuming 'model' is your QRCodeDetector model instance
model = QRCodeDetector()

# Load your state_dict
state_dict = torch.load('/home/surya/Downloads/alfaTKG/qr-scanner/ML/best.pt', map_location=torch.device('cpu'))

# Filter out unnecessary keys
filtered_state_dict = {k: v for k, v in state_dict.items() if k in model.state_dict()}

# Load filtered state_dict
model.load_state_dict(filtered_state_dict, strict=False)


# Instantiate the model
model = QRCodeDetector()

# Load the state dictionary
model_path = "/home/surya/Downloads/alfaTKG/qr-scanner/ML/best.pt"
state_dict = torch.load(model_path)
model.load_state_dict(state_dict)

# Set the model to evaluation mode
model.eval()

# Utility functions
def preprocess_image(image_path):
    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.ToTensor()
    ])
    image_tensor = transform(image_rgb).unsqueeze(0)
    return image, image_tensor

def detect_qr_codes(model, image_tensor):
    with torch.no_grad():
        predictions = model(image_tensor)
    return predictions

def postprocess(predictions, confidence_threshold=0.5):
    boxes = predictions[0]['boxes'].cpu().numpy()
    scores = predictions[0]['scores'].cpu().numpy()
    indices = np.where(scores >= confidence_threshold)[0]
    return boxes[indices], scores[indices]

def draw_boxes(image, boxes):
    for box in boxes:
        start_point = (int(box[0]), int(box[1]))
        end_point = (int(box[2]), int(box[3]))
        cv2.rectangle(image, start_point, end_point, (0, 255, 0), 2)
    return image

def extract_roi(image, box):
    x_min, y_min, x_max, y_max = map(int, box)
    return image[y_min:y_max, x_min:x_max]

def decode_qr_code(image):
    qr_data = qr_decode(image)
    if qr_data:
        return qr_data[0].data.decode("utf-8")
    return None

def main(image_path):
    # Preprocess the image
    original_image, image_tensor = preprocess_image(image_path)
    
    # Run inference
    predictions = detect_qr_codes(model, image_tensor)
    
    # Postprocess results
    boxes, scores = postprocess(predictions)
    
    # Draw bounding boxes on the image
    image_with_boxes = draw_boxes(original_image, boxes)
    
    # Extract and decode QR codes
    for box in boxes:
        roi = extract_roi(original_image, box)
        qr_code_data = decode_qr_code(roi)
        if qr_code_data:
            print("QR Code Data:", qr_code_data)
    
    # Display the image with detections
    cv2.imshow("QR Code Detection", image_with_boxes)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Run the main function
image_path = "/home/surya/Downloads/alfaTKG/qr-scanner/ML/1-14.png"
main(image_path)
