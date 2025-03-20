import cv2

# Load OpenCV’s pre-trained face detector
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

def face_detect(image_cv):
    # Convert to grayscale for face detection
    gray = cv2.cvtColor(image_cv, cv2.COLOR_RGB2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    if len(faces) <= 0:
        return None

    # Find the biggest face (by area)
    biggest_face = max(faces, key=lambda f: f[2] * f[3])  # w * h

    return biggest_face