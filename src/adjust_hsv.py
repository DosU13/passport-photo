import cv2
import numpy as np

from src.face_colors import extract_hsv_colors

# Define target levels
TARGET_BRIGHTNESS = 160
TARGET_CONTRAST = 40
TARGET_SATURATION = 78
TARGET_L = 168
TARGET_A = 140
TARGET_B = 143
TARGET_S = 78
TARGET_V = 192

def adjust_hsv(image):
    img = np.array(image)
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)

    h, s, v = cv2.split(hsv)
    face_s, face_v = extract_hsv_colors(img)

    face_s_mean, face_v_mean = np.mean(face_s), np.mean(face_v)
    s = np.clip(s * (TARGET_S / face_s_mean), 0, 255).astype(np.uint8)
    v = np.clip(v * (TARGET_V / face_v_mean), 0, 255).astype(np.uint8)

    hsv = cv2.merge([h, s, v])
    corrected_img = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)

    return corrected_img

