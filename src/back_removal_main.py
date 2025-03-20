import os
from PIL import Image

from src.back_removal import remove_background

input_folder = "Graphics"
output_folder = "Back Removed"

# Ensure the output folder exists
os.makedirs(output_folder, exist_ok=True)

# Process all images in the input folder
for filename in os.listdir(input_folder):
    if filename.lower().endswith(('.png', '.jpg', '.jpeg')):  # Process only images
        input_path = os.path.join(input_folder, filename)
        output_path = os.path.join(output_folder, filename)

        print(f"Processing: {filename}")

        # Open the image
        input_img = Image.open(input_path)

        final_image = remove_background(input_img)
        # Save the final image
        final_image.save(output_path)

        print(f"Saved: {output_path}")

print("Processing completed for all images.")

