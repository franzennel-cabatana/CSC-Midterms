"""
Step 1 - ENROLL the home owner.
Captures face samples from the webcam (or from a folder of photos) and saves them to
owners/<name>/. Run once per owner.

  python enroll_owner.py --name Juan                    # webcam
  python enroll_owner.py --name Juan --from-images pics # use existing photos

Webcam keys: SPACE = start/pause capturing | q = quit
Tip: vary head angle, expression and lighting while capturing for better accuracy.
"""
import argparse
import glob
import os
import time

import cv2
from face_utils import detect_faces, load_cascade, preprocess_face

PROMPTS = ["Look straight at the camera", "Turn slightly LEFT", "Turn slightly RIGHT",
           "Tilt head / change expression"]


def largest(boxes):
    return max(boxes, key=lambda b: b[2] * b[3])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--count", type=int, default=60, help="number of samples to capture")
    ap.add_argument("--source", default="0")
    ap.add_argument("--from-images", default="", help="folder of photos instead of webcam")
    args = ap.parse_args()

    out_dir = os.path.join("owners", args.name)
    os.makedirs(out_dir, exist_ok=True)
    cascade = load_cascade()
    saved = len(glob.glob(os.path.join(out_dir, "*.png")))

    # ---- Mode A: from existing photos ----
    if args.from_images:
        for path in sorted(glob.glob(os.path.join(args.from_images, "*"))):
            img = cv2.imread(path)
            if img is None:
                continue
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            boxes = detect_faces(cascade, gray)
            if len(boxes):
                cv2.imwrite(os.path.join(out_dir, f"{saved:03d}.png"),
                            preprocess_face(gray, largest(boxes)))
                saved += 1
        print(f"Saved {saved} face samples to {out_dir}")
        return

    # ---- Mode B: webcam ----
    cap = cv2.VideoCapture(int(args.source) if args.source.isdigit() else args.source)
    capturing, last_save, start_count = False, 0.0, saved
    while saved - start_count < args.count:
        ok, frame = cap.read()
        if not ok:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        boxes = detect_faces(cascade, gray)
        got = saved - start_count
        if len(boxes) == 1:                         # only save when exactly one face is visible
            x, y, w, h = boxes[0]
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 200, 0), 2)
            if capturing and time.time() - last_save > 0.2:
                cv2.imwrite(os.path.join(out_dir, f"{saved:03d}.png"),
                            preprocess_face(gray, boxes[0]))
                saved, last_save = saved + 1, time.time()
        prompt = PROMPTS[min(got * len(PROMPTS) // args.count, len(PROMPTS) - 1)]
        cv2.putText(frame, f"{args.name}: {got}/{args.count}  [{'CAPTURING' if capturing else 'PAUSED - press SPACE'}]",
                    (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        cv2.putText(frame, prompt, (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.imshow("Enroll owner", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        if key == ord(" "):
            capturing = not capturing
    cap.release()
    cv2.destroyAllWindows()
    print(f"Done. {saved - start_count} new samples in {out_dir}. Next: python train_recognizer.py")


if __name__ == "__main__":
    main()
