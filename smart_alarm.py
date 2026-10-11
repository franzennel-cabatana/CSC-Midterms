"""
Smart Intruder Alarm (owner vs intruder)  -  CSC 126 Midterm Project
====================================================================
Watches a camera feed, detects motion inside a restricted zone, and decides WHO it is:

  * Owner recognized  -> alarm suppressed, "OWNER: <name> - SAFE"
  * Unknown face      -> alarm "INTRUDER - UNKNOWN FACE"
  * Motion, no owner seen within the grace period -> alarm "UNIDENTIFIED MOTION"
    (covers an intruder whose face is hidden / turned away)

Lab topics
  (a) Image Operations ........ resize, grayscale, blur, morphology, histogram equalization
  (b) Image Segmentation ...... MOG2 background subtraction + threshold
  (c) Haar Cascade ............ face detection
  (d) Image Analysis .......... contours, area filter, bounding boxes
  (e) Motion & Object Tracking  centroid tracker (IDs + trails)
  + LBPH face recognition (cv2.face) to tell owner from stranger

Run:  python smart_alarm.py [--source 0 | video.mp4] [--threshold 60]
Keys: q quit | r re-learn background | a arm/disarm | s screenshot
"""
import argparse
import csv
import json
import os
import sys
import threading
import time
from collections import deque
from datetime import datetime

import cv2
import numpy as np

from face_utils import detect_faces, load_cascade, preprocess_face, require_face_module

try:
    import winsound
except ImportError:
    winsound = None

FRAME_WIDTH = 640
GREEN, RED, BLUE, YELLOW = (0, 200, 0), (0, 0, 255), (255, 120, 0), (0, 220, 255)


def sound_alarm():
    def _beep():
        if winsound:
            winsound.Beep(1800, 250)
        else:
            sys.stdout.write("\a")
            sys.stdout.flush()
    threading.Thread(target=_beep, daemon=True).start()


class CentroidTracker:
    """(e) Gives each moving object an ID and remembers its recent path."""

    def __init__(self, max_dist=80, max_missing=15):
        self.max_dist, self.max_missing, self.next_id = max_dist, max_missing, 0
        self.objects, self.missing, self.trails = {}, {}, {}

    def update(self, centroids):
        unmatched = list(range(len(centroids)))
        for oid, old in list(self.objects.items()):
            match = None
            if unmatched:
                d = [np.hypot(old[0] - centroids[i][0], old[1] - centroids[i][1]) for i in unmatched]
                j = int(np.argmin(d))
                if d[j] <= self.max_dist:
                    match = unmatched.pop(j)
            if match is not None:
                self.objects[oid], self.missing[oid] = centroids[match], 0
                self.trails[oid].append(centroids[match])
            else:
                self.missing[oid] += 1
                if self.missing[oid] > self.max_missing:
                    for store in (self.objects, self.missing, self.trails):
                        del store[oid]
        for i in unmatched:
            self.objects[self.next_id] = centroids[i]
            self.missing[self.next_id] = 0
            self.trails[self.next_id] = deque([centroids[i]], maxlen=30)
            self.next_id += 1
        return self.objects


def detect_motion(blur, subtractor, kernel, min_area, learning_rate):
    """(b)+(a)+(d): foreground mask -> cleaned -> contours above min_area."""
    fg = subtractor.apply(blur, learningRate=learning_rate)
    _, mask = cv2.threshold(fg, 200, 255, cv2.THRESH_BINARY)       # remove shadows (127)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.dilate(mask, kernel, iterations=2)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return mask, [c for c in contours if cv2.contourArea(c) >= min_area]


def inside(zone, pt):
    x, y, w, h = zone
    return x <= pt[0] <= x + w and y <= pt[1] <= y + h


def log_event(path, level, who):
    new = not os.path.exists(path)
    with open(path, "a", newline="") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["timestamp", "level", "who"])
        w.writerow([datetime.now().isoformat(timespec="seconds"), level, who])


def recognize(recognizer, labels, gray, box, threshold):
    """LBPH prediction. LOWER confidence value = BETTER match. Returns (name, conf, is_owner)."""
    label, conf = recognizer.predict(preprocess_face(gray, box))
    if conf < threshold:
        return labels.get(str(label), "owner"), conf, True
    return "UNKNOWN", conf, False


def main():
    ap = argparse.ArgumentParser(description="Smart intruder alarm with owner recognition")
    ap.add_argument("--source", default="0")
    ap.add_argument("--threshold", type=float, default=60.0,
                    help="LBPH distance below which a face counts as the owner (lower = stricter)")
    ap.add_argument("--min-area", type=int, default=1500)
    ap.add_argument("--entry-delay", type=float, default=3.0,
                    help="seconds of motion with no recognized owner before 'unidentified motion' alarm")
    ap.add_argument("--owner-grace", type=float, default=8.0,
                    help="seconds an owner sighting suppresses motion-only alarms")
    ap.add_argument("--unknown-frames", type=int, default=6,
                    help="consecutive frames with an unknown face before alarm")
    ap.add_argument("--face-every", type=int, default=2, help="run face check every N frames")
    ap.add_argument("--hold", type=float, default=3.0)
    ap.add_argument("--cooldown", type=float, default=10.0)
    ap.add_argument("--warmup", type=int, default=60)
    ap.add_argument("--no-zone", action="store_true")
    ap.add_argument("--save-video", default="")
    args = ap.parse_args()

    require_face_module()
    if not (os.path.exists("model.yml") and os.path.exists("labels.json")):
        sys.exit("No trained model. Run:  python enroll_owner.py --name YOU  then  python train_recognizer.py")
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read("model.yml")
    with open("labels.json") as f:
        labels = json.load(f)
    cascade = load_cascade()

    source = int(args.source) if args.source.isdigit() else args.source
    cap = cv2.VideoCapture(source)
    ok, frame = cap.read()
    if not ok:
        sys.exit(f"Could not open video source: {args.source}")
    frame = cv2.resize(frame, (FRAME_WIDTH, int(frame.shape[0] * FRAME_WIDTH / frame.shape[1])))
    H, W = frame.shape[:2]

    zone = (0, 0, W, H)
    if not args.no_zone:
        r = cv2.selectROI("Select restricted zone (ENTER to confirm)", frame, False)
        cv2.destroyWindow("Select restricted zone (ENTER to confirm)")
        if r[2] > 0 and r[3] > 0:
            zone = tuple(int(v) for v in r)

    subtractor = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=50, detectShadows=True)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    tracker = CentroidTracker()
    writer = (cv2.VideoWriter(args.save_video, cv2.VideoWriter_fourcc(*"mp4v"), 20, (W, H))
              if args.save_video else None)
    os.makedirs("snapshots", exist_ok=True)

    armed, frame_no = True, 0
    faces_info = []                                  # [(box, name, conf, is_owner), ...]
    unknown_streak = 0
    motion_since, last_motion = None, 0.0
    owner_until, owner_name = 0.0, ""
    alarm_until, last_event, last_beep = 0.0, 0.0, 0.0
    level, alarm_kind = "CLEAR", ""
    prev_t, fps = time.time(), 0.0

    while True:
        if frame_no > 0:
            ok, frame = cap.read()
            if not ok:
                break
            frame = cv2.resize(frame, (W, H))
        frame_no += 1
        now = time.time()
        warming = frame_no <= args.warmup

        # (a) preprocessing
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (21, 21), 0)

        # (b)(d) motion segmentation + contours
        mask, contours = detect_motion(blur, subtractor, kernel, args.min_area,
                                       -1 if warming else 0.002)
        boxes, centroids = [], []
        for c in contours:
            x, y, w, h = cv2.boundingRect(c)
            boxes.append((x, y, w, h))
            centroids.append((x + w // 2, y + h // 2))
        objects = {} if warming else tracker.update(centroids)
        motion_in_zone = (not warming) and any(inside(zone, c) for c in centroids)

        # (c) + recognition: who is in the picture? (every N frames, result cached between)
        if warming:
            faces_info = []
        elif frame_no % args.face_every == 0:
            faces_info = []
            for box in detect_faces(cascade, gray):
                name, conf, is_owner = recognize(recognizer, labels, gray, box, args.threshold)
                faces_info.append((tuple(int(v) for v in box), name, conf, is_owner))

        zone_faces = [f for f in faces_info
                      if inside(zone, (f[0][0] + f[0][2] // 2, f[0][1] + f[0][3] // 2))]
        owners_now = [f for f in zone_faces if f[3]]
        unknowns_now = [f for f in zone_faces if not f[3]]

        if owners_now:                               # owner sighting -> start/refresh grace period
            owner_until, owner_name = now + args.owner_grace, owners_now[0][1]
        owner_recent = now < owner_until

        unknown_streak = unknown_streak + 1 if unknowns_now else 0

        if motion_in_zone:
            last_motion = now
            motion_since = motion_since or now
        elif motion_since and now - last_motion > 1.0:
            motion_since = None

        # ---------------- decision logic ----------------
        unknown_alarm = unknown_streak >= args.unknown_frames
        motion_alarm = (motion_since is not None and now - motion_since >= args.entry_delay
                        and not owner_recent)
        if armed and (unknown_alarm or motion_alarm):
            alarm_kind = "INTRUDER - UNKNOWN FACE" if unknown_alarm else "UNIDENTIFIED MOTION"
            alarm_until = now + args.hold
            if now - last_beep >= 0.4:
                last_beep = now
                sound_alarm()
            if now - last_event >= args.cooldown:
                last_event = now
                cv2.imwrite(f"snapshots/intruder_{datetime.now():%Y%m%d_%H%M%S}.jpg", frame)
                log_event("alarm_log.csv", alarm_kind, "unknown" if unknown_alarm else "no face")
        alarm_on = armed and now < alarm_until

        if alarm_on:
            level = alarm_kind
        elif warming:
            level = "WARMING UP (stay out of view)"
        elif not armed:
            level = "DISARMED"
        elif owner_recent:
            level = f"OWNER: {owner_name} - SAFE"
        else:
            level = "CLEAR"

        # ---------------- visualization ----------------
        out = frame.copy()
        zx, zy, zw, zh = zone
        cv2.rectangle(out, (zx, zy), (zx + zw, zy + zh), RED if alarm_on else GREEN, 2)
        for (x, y, w, h) in boxes:
            cv2.rectangle(out, (x, y), (x + w, y + h), YELLOW, 1)
        for (x, y, w, h), name, conf, is_owner in faces_info:
            col = GREEN if is_owner else RED
            cv2.rectangle(out, (x, y), (x + w, y + h), col, 2)
            cv2.putText(out, f"{name} ({conf:.0f})", (x, y - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, col, 2)
        for oid, (cx, cy) in objects.items():
            cv2.circle(out, (cx, cy), 4, BLUE, -1)
            cv2.putText(out, f"ID {oid}", (cx + 6, cy - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.45, BLUE, 1)
            cv2.polylines(out, [np.array(tracker.trails[oid], np.int32).reshape(-1, 1, 2)],
                          False, BLUE, 1)

        banner = RED if alarm_on else (GREEN if armed else (128, 128, 128))
        cv2.rectangle(out, (0, 0), (W, 28), banner, -1)
        cv2.putText(out, f"STATUS: {level}", (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (255, 255, 255), 2)
        fps = 0.9 * fps + 0.1 / max(now - prev_t, 1e-6)
        prev_t = now
        cv2.putText(out, f"{datetime.now():%Y-%m-%d %H:%M:%S}  FPS {fps:.0f}  "
                         f"{'ARMED' if armed else 'DISARMED'}  thr={args.threshold:.0f}",
                    (8, H - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        cv2.imshow("Smart Intruder Alarm", out)
        cv2.imshow("Motion Mask (segmentation)", mask)
        if writer:
            writer.write(out)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("a"):
            armed, unknown_streak, motion_since = not armed, 0, None
        elif key == ord("r"):
            subtractor = cv2.createBackgroundSubtractorMOG2(500, 50, True)
            tracker, frame_no, unknown_streak, motion_since = CentroidTracker(), 1, 0, None
        elif key == ord("s"):
            cv2.imwrite(f"snapshots/screenshot_{datetime.now():%H%M%S}.png", out)

    cap.release()
    if writer:
        writer.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
