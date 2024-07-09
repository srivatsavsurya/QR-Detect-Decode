import cv2
from pyzbar.pyzbar import decode
from ultralytics import YOLO
import json
import os

yolo_model = YOLO('best.pt')

def detect_qr_codes_in_image(image_path, margin=10):
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Image not found at path: {image_path}")

    height, width, _ = img.shape

    results = yolo_model(img)
    boxes = results[0].boxes 
    print("Number of boxes detected:", len(boxes))

    qr_codes = []

    for i, box in enumerate(boxes):
        try:
            # Move the tensor to CPU and convert to numpy array
            x, y, x1, y1 = box.xyxy[0].cpu().numpy()  # Get bounding box coordinates

            # Add margin around the bounding box
            x = max(0, x - margin)
            y = max(0, y - margin)
            x1 = min(width, x1 + margin)
            y1 = min(height, y1 + margin)

            print(f"Bounding box coordinates with margin: x={x}, y={y}, x1={x1}, y1={y1}")

            # Crop and process the region containing the detected object (QR code in this case)
            qr_area = img[int(y):int(y1), int(x):int(x1)]

            # Save the cropped area for inspection
            cropped_image_path = f'cropped_qr_{i}.jpg'
            cv2.imwrite(cropped_image_path, qr_area)
            print(f"Cropped QR area saved to: {cropped_image_path}")

            # Decode QR codes using PyZbar
            decoded_objects = decode(qr_area)
            print("Decoded objects:", decoded_objects)

            # Extract information from decoded QR codes
            for obj in decoded_objects:
                data = obj.data.decode('utf-8')
                qr_codes.append({
                    "data": data,
                    "type": obj.type,
                    "rect": {
                        "x": int(x), "y": int(y), "width": int(x1 - x), "height": int(y1 - y)
                    }
                })

        except Exception as e:
            print(f"Error processing box {i}: {e}")

    return qr_codes

# Example usage
if __name__ == '__main__':
    image_path = '2024-07-09006_page-0001.jpg'  # Replace with your image path

    try:
        qr_codes = detect_qr_codes_in_image(image_path, margin=20)  # Adjust margin as needed
        json_output = json.dumps(qr_codes, indent=4)
        print(json_output)
    except Exception as e:
        print(f"Error detecting QR codes: {e}")
