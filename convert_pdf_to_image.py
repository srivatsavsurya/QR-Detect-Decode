import os
from pdf2image import convert_from_path
from PIL import Image, ImageEnhance

# Input and output folder paths
input_folder = 'PDFs'
output_folder = 'Images'

def convert_pdf_to_image(pdf_path, output_folder):
    # Ensure the output folder exists
    os.makedirs(output_folder, exist_ok=True)

    # Convert PDF to images
    images = convert_from_path(pdf_path, dpi=300)  # Using 300 DPI for high resolution

    for i, image in enumerate(images):
        # Convert the image to 4K resolution
        image_4k = image.resize((3840, 2160), Image.Resampling.LANCZOS)

        # Enhance the image to simulate HDR
        enhancer = ImageEnhance.Brightness(image_4k)
        image_hdr = enhancer.enhance(1.2)  # Adjust brightness
        enhancer = ImageEnhance.Contrast(image_hdr)
        image_hdr = enhancer.enhance(1.3)  # Adjust contrast

        # Save the image
        base_name = os.path.splitext(os.path.basename(pdf_path))[0]
        output_path = os.path.join(output_folder, f'{base_name}_page_{i + 1}.png')
        image_hdr.save(output_path, format='PNG')

        print(f'Saved: {output_path}')

def process_folder(input_folder, output_folder):
    # Ensure the output folder exists
    os.makedirs(output_folder, exist_ok=True)

    # Process each PDF in the input folder
    for file_name in os.listdir(input_folder):
        if file_name.lower().endswith('.pdf'):
            pdf_path = os.path.join(input_folder, file_name)
            convert_pdf_to_image(pdf_path, output_folder)

if __name__ == "__main__":
    process_folder(input_folder, output_folder)
