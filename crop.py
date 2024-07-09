import cv2
import numpy as np
import pyboof as pb
from pyzbar.pyzbar import decode
import json

def detect_and_decode_qr(image_path, margin=10):
    # Load image, grayscale, Gaussian blur, Otsu's threshold
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Image not found at path: {image_path}")
    original = image.copy()
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (9, 9), 0)
    thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

    # Morph close
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    close = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)

    # Find contours and filter for QR code
    cnts = cv2.findContours(close, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]

    qr_codes = []

    for i, c in enumerate(cnts):
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.04 * peri, True)
        x, y, w, h = cv2.boundingRect(approx)
        area = cv2.contourArea(c)
        ar = w / float(h)
        if len(approx) == 4 and area > 1000 and (ar > 0.85 and ar < 1.3):
            x = max(0, x - margin)
            y = max(0, y - margin)
            x1 = min(image.shape[1], x + w + margin)
            y1 = min(image.shape[0], y + h + margin)

            cv2.rectangle(image, (x, y), (x1, y1), (36, 255, 12), 3)
            ROI = original[y:y1, x:x1]

            # Save the cropped QR area for inspection
            cropped_image_path = f'cropped_qr_{i}.jpg'
            cv2.imwrite(cropped_image_path, ROI)
            print(f"Cropped QR area saved to: {cropped_image_path}")

            # Decode QR codes using PyZbar
            decoded_objects = decode(ROI)
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

    return qr_codes

# Example usage
if __name__ == '__main__':
    image_path = '/home/surya/Downloads/alfaTKG/qr-scanner/ML/2024-07-09006_page-0001.jpg'  # Replace with your image path

    try:
        qr_codes = detect_and_decode_qr(image_path, margin=20)  # Adjust margin as needed
        json_output = json.dumps(qr_codes, indent=4)
        print(json_output)

    except Exception as e:
        print(f"Error detecting QR codes: {e}")
