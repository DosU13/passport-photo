import os
import numpy as np
from PIL import Image

from src.resize_image import resize_image

# Define input and output folders
input_folder = "Back Removed"
output_folder = "Face Detect"

# Ensure the output folder exists
os.makedirs(output_folder, exist_ok=True)

# Process all images in the input folder
for filename in os.listdir(input_folder):
    if filename.lower().endswith(('.png', '.jpg', '.jpeg')):  # Process only images
        input_path = os.path.join(input_folder, filename)
        output_path = os.path.join(output_folder, filename)

        # Load the image
        image = Image.open(input_path)
        image_cv = np.array(image)  # Convert to OpenCV format (NumPy array)

        final_image = resize_image(image, image_cv)
        # Save the final processed image
        final_image.save(output_path)

print("Face detection & cropping completed for all images.")
