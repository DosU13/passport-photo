import os
import numpy as np
from PIL import Image

from src.adjust_hsv import adjust_hsv

input_folder = "Face Detect"
output_folder = "Corrected"

os.makedirs(output_folder, exist_ok=True)

# Process images
for filename in os.listdir(input_folder):
    if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
        input_path = os.path.join(input_folder, filename)
        output_path = os.path.join(output_folder, filename)

        # Load image
        image = Image.open(input_path).convert("RGB")
        np_image = np.array(image)

        # Apply unified brightness, contrast, and saturation correction
        corrected_image = adjust_hsv(np_image)

        # Save corrected image
        Image.fromarray(corrected_image).save(output_path)

print("Image correction completed. Check the 'Corrected' folder.")
