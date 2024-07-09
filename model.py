import cv2
from pyzbar.pyzbar import decode
import tensorflow as tf
import numpy as np
import json

# Function to load and use the QR code detection model
def load_qr_detection_model(model_path):
    # Example using TensorFlow SavedModel format
    model = tf.saved_model.load(model_path)
    return model

# Function to detect QR codes using the loaded model
def detect_qr_codes_with_model(model, image_path):
    # Load the image
    img = cv2.imread(image_path)
    
    # Convert to grayscale (if needed)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Preprocess the image as needed for your model
    processed_img = preprocess_image(gray)  # Define this function based on your model's requirements
    
    # Run inference
    predictions = model(processed_img)
    
    # Postprocess predictions to extract QR code information
    qr_codes = extract_qr_codes(predictions)
    
    return qr_codes

# Function to preprocess image data for the model
def preprocess_image(image):
    # Example preprocessing: resize, normalize, etc.
    resized_img = cv2.resize(image, (224, 224))
    normalized_img = resized_img / 255.0  # Normalize to [0, 1]
    return np.expand_dims(normalized_img, axis=0)  # Add batch dimension

# Function to extract QR code information from model predictions
def extract_qr_codes(predictions):
    # Example: Decode QR codes from predictions
    # This depends heavily on your specific model's output format and postprocessing
    decoded_objects = decode(predictions)  # Example using PyZbar for QR code decoding
    
    qr_codes = []
    for obj in decoded_objects:
        data = obj.data.decode('utf-8')
        qr_codes.append({"data": data, "type": obj.type, "rect": obj.rect})
    
    return qr_codes

# Example usage
if __name__ == '__main__':
    model_path = '/home/surya/Downloads/alfaTKG/qr-scanner/ML/best.pt'  # Replace with your model path
    image_path = '/home/surya/Downloads/alfaTKG/qr-scanner/ML/1-14.png'  # Replace with your image path
    
    model = load_qr_detection_model(model_path)
    qr_codes = detect_qr_codes_with_model(model, image_path)
    
    # Output results as JSON
    json_output = json.dumps(qr_codes, indent=4)
    print(json_output)
