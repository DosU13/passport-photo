import os
import cv2
import numpy as np
import pandas as pd
from PIL import Image

from src.face_colors import extract_colors, extract_lab_colors, extract_hsv_colors
from src.resize_image import find_hairline

# Define input folder and output file
input_folder = "Corrected"
output_file = "Corrected.csv"

# Ensure the input folder exists
if not os.path.exists(input_folder):
    print(f"Error: Folder '{input_folder}' not found.")
    exit()

# Load OpenCV’s pre-trained face detector
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# Analyze images
data = []
for filename in os.listdir(input_folder):
    if filename.lower().endswith(('.png', '.jpg', '.jpeg')):  # Process only images
        file_path = os.path.join(input_folder, filename)
        image = Image.open(file_path).convert("RGB")
        image_cv = np.array(image)

        # Convert to grayscale for face detection
        gray = cv2.cvtColor(image_cv, cv2.COLOR_RGB2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        # If no face detected, skip the image
        if len(faces) == 0:
            print(f"No face detected in {filename}, skipping.")
            continue

        # Find the biggest face (by area)
        biggest_face = max(faces, key=lambda f: f[2] * f[3])
        (x, y, w, h) = biggest_face

        # Estimate hairline (topest non-white pixel)
        hairline_y = find_hairline(image_cv)

        if hairline_y is None:
            hairline_y = y  # Default to top of the face

        # Calculate margins
        top_margin = hairline_y  # Pixels from top to hairline
        bottom_margin = image_cv.shape[0] - (y + h)  # Pixels from chin to bottom

        # Calculate image properties
        (s, v) = extract_hsv_colors(image_cv)

        # Store data
        data.append([filename, s, v, top_margin, bottom_margin])

# Convert to DataFrame
df = pd.DataFrame(data, columns=["Filename", "S", "V", "Top Margin", "Bottom Margin"])

# Calculate averages
averages = df.mean(numeric_only=True).to_dict()
averages["Filename"] = "Averages"
df = pd.concat([df, pd.DataFrame([averages])], ignore_index=True)

# Save results to CSV
df.to_csv(output_file, index=False)
print(f"Analysis completed. Results saved to '{output_file}'.")
