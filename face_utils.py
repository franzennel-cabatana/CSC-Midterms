"""Shared helpers: Haar face detection + face preprocessing (used by all scripts)."""
import sys
import cv2

FACE_SIZE = (200, 200)       # every face crop is resized to this before training/predicting


def load_cascade():
    """(c) Load OpenCV's pretrained frontal-face Haar cascade."""
    cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    if cascade.empty():
        sys.exit("Failed to load Haar cascade.")
    return cascade


def detect_faces(cascade, gray, min_size=(60, 60)):
    """Return list of (x, y, w, h) face boxes found in a grayscale image."""
    return cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=min_size)


def preprocess_face(gray, box):
    """(a) Crop -> resize -> histogram equalization (reduces lighting differences)."""
    x, y, w, h = box
    face = cv2.resize(gray[y:y + h, x:x + w], FACE_SIZE)
    return cv2.equalizeHist(face)


def require_face_module():
    if not hasattr(cv2, "face"):
        sys.exit("cv2.face not found. Run:\n  pip uninstall opencv-python\n"
                 "  pip install opencv-contrib-python")
