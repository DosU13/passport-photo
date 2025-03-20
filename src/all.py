import os
import cv2

from PIL import Image

from src.adjust_hsv import adjust_hsv
from src.back_removal import remove_background
from src.resize_image import resize_image

input_folder = "Graphics"
output_folder = "My Graphics Id"

# Ensure the output folder exists
os.makedirs(output_folder, exist_ok=True)

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# Target resolution
TARGET_WIDTH = 240
TARGET_HEIGHT = 320
TARGET_RATIO = TARGET_WIDTH / TARGET_HEIGHT  # Aspect ratio 0.75
TOP_MARGIN = 15
BOTTOM_MARGIN = 83

for filename in os.listdir(input_folder):
    if filename.lower().endswith(('.png', '.jpg', '.jpeg')):  # Process only images
        input_path = os.path.join(input_folder, filename)
        output_path = os.path.join(output_folder, filename)

        print(f"Processing: {filename}")

        # Open the image
        image = Image.open(input_path)

        image = remove_background(image)

        image = resize_image(image)

        image_sv = adjust_hsv(image)

        Image.fromarray(image_sv).save(output_path)

print("Processing completed for all images.")


