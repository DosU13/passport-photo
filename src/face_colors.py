import cv2
import numpy as np
from PIL import Image

# Load OpenCV’s pre-trained face detector
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")


def extract_face(image_cv):
    """
    Detects and extracts the biggest face from the image while avoiding hair and clothing.
    Returns the cropped face or None if no face is detected.
    """
    gray = cv2.cvtColor(image_cv, cv2.COLOR_RGB2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    if len(faces) == 0:
        print("No face detected")
        return None

    (x, y, w, h) = max(faces, key=lambda f: f[2] * f[3])
    face_only = image_cv[y + int(0.2 * h): y + int(0.8 * h), x + int(0.1 * w): x + int(0.9 * w)]

    return face_only if face_only.size != 0 else None


def face_brightness(face):
    """
    Calculates the brightness of the given face image.
    """
    return np.mean(face) if face is not None else None


def face_contrast(face):
    """
    Calculates the contrast of the given face image using standard deviation.
    """
    return np.std(face) if face is not None else None


def face_saturation(face):
    """
    Calculates the saturation of the given face image in the HSV color space.
    """
    if face is None:
        return None
    hsv = cv2.cvtColor(face, cv2.COLOR_RGB2HSV)
    return np.mean(hsv[:, :, 1])  # Mean saturation value

def extract_colors(image_cv):

    face = extract_face(image_cv)
    bridge_color = face_brightness(face)
    contrast_color = face_contrast(face)
    saturation_color = face_saturation(face)
    return bridge_color, contrast_color, saturation_color

def extract_lab_colors(image_cv):
    face = extract_face(image_cv)

    lab = cv2.cvtColor(face, cv2.COLOR_RGB2LAB)
    # Split LAB channels
    l, a, b = cv2.split(lab)

    l_mean, a_mean, b_mean = np.mean(l), np.mean(a), np.mean(b)

    return l_mean, a_mean, b_mean

def extract_hsv_colors(image_cv):
    face = extract_face(image_cv)

    hsv = cv2.cvtColor(face, cv2.COLOR_RGB2HSV)
    h, s, v = cv2.split(hsv)

    s_mean, v_mean = np.mean(s), np.mean(v)

    return s_mean, v_mean