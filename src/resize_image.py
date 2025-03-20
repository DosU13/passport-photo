import cv2
import numpy as np
from PIL import Image

from src.face_detect import face_detect
from src.find_hairline import find_hairline

# Target resolution
TARGET_WIDTH = 240
TARGET_HEIGHT = 320
TARGET_RATIO = TARGET_WIDTH / TARGET_HEIGHT  # Aspect ratio 0.75
TOP_MARGIN = 15
BOTTOM_MARGIN = 83

def resize_image(image):
    image_cv = np.array(image)
    (x, y, w, h) = face_detect(image_cv)

    # Get image dimensions
    img_h, img_w, _ = image_cv.shape
    horizontal_mid = x + w // 2  # Center of the face
    chin = y + h  # Bottom of the detected face
    hairline = find_hairline(image_cv)  # Find the top of the head

    # Adjust top and bottom positions
    face_height = chin - hairline
    top = max(0, hairline - int(face_height * (TOP_MARGIN / (TARGET_HEIGHT - BOTTOM_MARGIN))))
    bottom = min(img_h, chin + int(face_height * (BOTTOM_MARGIN / (TARGET_HEIGHT - TOP_MARGIN))))

    # Compute new height
    new_height = bottom - top
    new_width = int(new_height * TARGET_RATIO)  # Maintain aspect ratio 0.75

    # Ensure width is within bounds and centered
    left = max(0, horizontal_mid - new_width // 2)
    right = min(img_w, left + new_width)

    # Crop the image
    cropped_image = image.crop((left, top, right, bottom))

    # Resize to 240x320 pixels
    final_image = cropped_image.resize((TARGET_WIDTH, TARGET_HEIGHT), Image.LANCZOS)
    return final_image
