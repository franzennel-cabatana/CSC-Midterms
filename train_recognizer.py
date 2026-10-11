"""
Step 2 - TRAIN the LBPH face recognizer on everything in owners/<name>/*.png.
Outputs model.yml (the trained model) and labels.json (label id -> owner name).

  python train_recognizer.py
"""
import glob
import json
import os
import sys

import cv2
import numpy as np
from face_utils import require_face_module


def main():
    require_face_module()
    images, labels, names = [], [], {}
    for label, person in enumerate(sorted(os.listdir("owners")) if os.path.isdir("owners") else []):
        files = glob.glob(os.path.join("owners", person, "*.png"))
        if not files:
            continue
        names[label] = person
        for f in files:
            img = cv2.imread(f, cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            images.append(img)
            labels.append(label)
            images.append(cv2.flip(img, 1))          # augmentation: mirrored copy
            labels.append(label)
        print(f"{person}: {len(files)} samples")

    if len(images) < 20:
        sys.exit("Not enough samples. Run enroll_owner.py first (aim for 50+ per owner).")

    # LBPH = Local Binary Patterns Histograms (classical, no deep learning)
    recognizer = cv2.face.LBPHFaceRecognizer_create(radius=1, neighbors=8, grid_x=8, grid_y=8)
    recognizer.train(images, np.array(labels))
    recognizer.write("model.yml")
    with open("labels.json", "w") as f:
        json.dump({str(k): v for k, v in names.items()}, f)
    print(f"Trained on {len(images)} images -> model.yml, labels.json")


if __name__ == "__main__":
    main()
