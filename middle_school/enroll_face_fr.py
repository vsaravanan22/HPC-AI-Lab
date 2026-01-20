#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Auto-enroll faces with Intel RealSense + OpenCV SSD + face_recognition.

Workflow:
- Start the script with your --person-id and --name (or pass via prompt).
- Press 'c' ONCE to begin auto-capture. The app will collect N samples (default 100)
  while you rotate your head slowly in a CLOCKWISE circle. Try a range of angles,
  head tilts, expressions, and with/without glasses.
- A progress bar shows how many samples have been recorded.
- When finished, it saves all embeddings into a single .npy file and updates:
    - labels.json  (maps your display name to the .npy file)
    - roster.csv   (id,name)

Hotkeys:
  c  start / restart auto-capture
  s  save immediately (even if not at N samples yet)
  r  reset collected samples to 0
  p  snapshot PNG to /home/alpha/realsense_enroll.png
  q/ESC  quit

Tips for higher recognition confidence later:
- Capture diverse angles and lighting.
- Include both WITH and WITHOUT glasses if you wear them.
- Keep your face mostly filling the box (not too small), keep camera steady.
- During recognition runs, you may slightly relax tolerance (e.g., 0.50–0.60).
"""
import os
import sys
import time
import json
import csv
import math
import argparse
from pathlib import Path
from typing import List, Tuple, Dict

import numpy as np
import cv2
import pyrealsense2 as rs
import face_recognition
import board
import busio
from adafruit_pn532.i2c import PN532_I2C

# ---------------- Paths ----------------
DETECT_PROTO = "/home/alpha/face_models/deploy.prototxt"
DETECT_MODEL = "/home/alpha/face_models/res10_300x300_ssd_iter_140000.caffemodel"

EMBED_DIR   = "/home/alpha/embeddings"
LABELS_PATH = "/home/alpha/embeddings/labels.json"
ROSTER_PATH = "/home/alpha/roster.csv"

# ---------------- Utils ----------------
def ensure_parent(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)

def load_labels(path: Path) -> Dict[str, str]:
    if path.exists():
        try:
            with path.open("r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, dict):
                return {str(k): str(v) for k, v in data.items()}
        except Exception:
            pass
    return {}

def save_labels(path: Path, labels: Dict[str, str]):
    ensure_parent(path)
    with path.open("w", encoding="utf-8") as f:
        json.dump(labels, f, indent=2)

def ensure_roster_header(path: Path):
    ensure_parent(path)
    if not path.exists():
        with path.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=["id","name", "card_uid"])
            w.writeheader()

def load_roster(path: Path) -> Dict[str, str]:
    d = {}
    if path.exists():
        with path.open("r", newline="", encoding="utf-8") as f:
            r = csv.DictReader(f)
            for row in r:
                pid = (row.get("id") or "").strip()
                nm  = (row.get("name") or "").strip()
                if pid:
                    d[pid] = nm
    return d

def upsert_roster(path: Path, person_id: str, person_name: str, card_uid: str | None):
    ensure_roster_header(path)

    rows = []
    found = False

    if path.exists():
        with path.open("r", newline="", encoding="utf-8") as f:
            r = csv.DictReader(f)
            for row in r:
                if row["id"] == person_id:
                    # update existing
                    rows.append({
                        "id": person_id,
                        "name": person_name,
                        "card_uid": card_uid or row.get("card_uid", "")
                    })
                    found = True
                else:
                    rows.append(row)

    if not found:
        rows.append({
            "id": person_id,
            "name": person_name,
            "card_uid": card_uid or ""
        })

    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["id", "name", "card_uid"])
        w.writeheader()
        for row in rows:
            w.writerow(row)

def expand_box(x1,y1,x2,y2, img_w, img_h, scale=0.3):
    w = x2 - x1
    h = y2 - y1
    cx = x1 + w//2
    cy = y1 + h//2
    new_w = int(w * (1.0 + scale))
    new_h = int(h * (1.0 + scale))
    nx1 = max(0, cx - new_w//2)
    ny1 = max(0, cy - new_h//2)
    nx2 = min(img_w-1, cx + new_w//2)
    ny2 = min(img_h-1, cy + new_h//2)
    return nx1, ny1, nx2, ny2

def detect_faces(net, frame_bgr, conf_thresh=0.5):
    (h, w) = frame_bgr.shape[:2]
    blob = cv2.dnn.blobFromImage(frame_bgr, 1.0, (300, 300), (104.0, 177.0, 123.0), swapRB=False, crop=False)
    net.setInput(blob)
    det = net.forward()
    boxes = []
    if det.ndim == 4 and det.shape[2] > 0:
        for i in range(det.shape[2]):
            conf = float(det[0, 0, i, 2])
            if conf < conf_thresh:
                continue
            x1 = int(det[0, 0, i, 3] * w)
            y1 = int(det[0, 0, i, 4] * h)
            x2 = int(det[0, 0, i, 5] * w)
            y2 = int(det[0, 0, i, 6] * h)
            x1 = max(0, min(x1, w-1))
            y1 = max(0, min(y1, h-1))
            x2 = max(0, min(x2, w-1))
            y2 = max(0, min(y2, h-1))
            if x2 > x1 and y2 > y1:
                boxes.append((x1,y1,x2,y2,conf))
    return boxes

def get_nfc_card_uid():
    print("\n[PN532] Initializing NFC reader...")

    # Initialize I2C
    try:
        i2c = busio.I2C(board.SCL, board.SDA)
    except Exception as e:
        print(f"[PN532] ERROR: Could not initialize I2C: {e}")
        return None

    # Initialize PN532
    try:
        pn532 = PN532_I2C(i2c, debug=False)
    except Exception as e:
        print(f"[PN532] ERROR: Could not initialize PN532: {e}")
        return None

    # Get firmware (optional, informative)
    try:
        ic, ver, rev, support = pn532.firmware_version
        print(f"[PN532] Found PN532 with firmware version {ver}.{rev}")
    except Exception as e:
        print(f"[PN532] ERROR: Could not read firmware: {e}")
        return None

    pn532.SAM_configuration()
    print("[PN532] Tap a card to assign to this student...")

    # Wait for a card
    import time
    start = time.time()
    timeout_seconds = 30  # change if you want shorter/longer

    while True:
        uid = pn532.read_passive_target(timeout=0.5)

        if uid is not None:
            # Convert UID bytes to hex string
            return "".join(f"{b:02X}" for b in uid)

        # Timeout
        if time.time() - start > timeout_seconds:
            print("[PN532] Timeout waiting for card.")
            return None

# ---------------- Main ----------------
def main():
    ap = argparse.ArgumentParser(description="Auto Enroll with RealSense + face_recognition")
    ap.add_argument("--det-prototxt", default=DETECT_PROTO)
    ap.add_argument("--det-caffemodel", default=DETECT_MODEL)
    ap.add_argument("--embeddings", default=EMBED_DIR)
    ap.add_argument("--labels", default=LABELS_PATH)
    ap.add_argument("--roster", default=ROSTER_PATH)
    ap.add_argument("--person-id", default="", help="Your unique ID (e.g., student/employee #)")
    ap.add_argument("--name", default="", help="Display name to store")
    ap.add_argument("--samples", type=int, default=1000, help="Number of embeddings to collect")
    ap.add_argument("--interval", type=float, default=0.001, help="Min seconds between samples")
    ap.add_argument("--min-enc-shift", type=float, default=0.03, help="Min encoding distance from last sample")
    ap.add_argument("--det-conf", type=float, default=0.70, help="Detector confidence threshold")
    ap.add_argument("--jitter", type=int, default=1, help="face_recognition num_jitters for enrollment encodings")
    ap.add_argument("--handsfree", action="store_true", help="Begin capture immediately on start")
    args = ap.parse_args()

    emb_dir   = Path(args.embeddings)
    labels_p  = Path(args.labels)
    roster_p  = Path(args.roster)

    # Ask for missing identity info
    person_id = args.person_id.strip()
    person_name = args.name.strip()
    if not person_id:
        person_id = input("Enter student ID: ").strip()
    if not person_name:
        person_name = input("Enter student name: ").strip()
    if not person_id or not person_name:
        print("[ERROR] ID and Name are required.")
        sys.exit(1)
    print("[ENROLL] Assigning NFC card...")
    card_uid = get_nfc_card_uid()

    if card_uid:
        print(f"[ENROLL] Card UID assigned: {card_uid}")
    else:
        print("[ENROLL] WARNING: No card assigned.")

    # Prepare outputs
    ensure_parent(emb_dir)
    ensure_parent(labels_p)
    ensure_roster_header(roster_p)
    out_npy = emb_dir / f"{person_name}.npy"

    # Load/prepare detector
    if not Path(args.det_prototxt).exists() or not Path(args.det_caffemodel).exists():
        print("[ERROR] Detector model files not found.")
        sys.exit(1)
    net = cv2.dnn.readNetFromCaffe(str(args.det_prototxt), str(args.det_caffemodel))

    # RealSense
    pipeline = rs.pipeline()
    cfg = rs.config()
    cfg.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)
    profile = pipeline.start(cfg)

    try:
        collected: List[np.ndarray] = []
        last_sample_ts = 0.0
        last_enc = None
        capturing = args.handsfree

        print("[INFO] Press 'c' to start auto-capture, 'q' to quit.")
        while True:
            frames = pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()
            if not color_frame:
                continue
            frame = np.asanyarray(color_frame.get_data())
            disp = frame.copy()
            h, w = frame.shape[:2]

            # Detect face
            boxes = detect_faces(net, frame, conf_thresh=args.det_conf)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Draw and (optionally) capture
            face_present = False
            if boxes:
                # Choose the largest face (by area)
                (x1,y1,x2,y2,conf) = max(boxes, key=lambda b: (b[2]-b[0])*(b[3]-b[1]))
                ex1,ey1,ex2,ey2 = expand_box(x1,y1,x2,y2, w,h, scale=0.25)
                cv2.rectangle(disp, (ex1,ey1), (ex2,ey2), (0,200,0), 2)
                face_present = True

                # Compute encoding
                loc = [(ey1, ex2, ey2, ex1)]  # top, right, bottom, left
                encs = face_recognition.face_encodings(rgb, known_face_locations=loc, num_jitters=args.jitter)
                if encs:
                    enc = encs[0]
                    now = time.time()
                    # Gate samples: time interval + encoding shift vs last_enc
                    ok_time = (now - last_sample_ts) >= args.interval
                    ok_shift = True
                    if last_enc is not None:
                        dist = np.linalg.norm(enc - last_enc)
                        ok_shift = dist >= args.min_enc_shift
                    if capturing and ok_time and ok_shift:
                        collected.append(enc.astype("float64"))
                        last_sample_ts = now
                        last_enc = enc
                        # visual tick
                        cv2.circle(disp, (ex1+10, ey1+10), 6, (0,200,0), -1)

            # HUD
            N = len(collected)
            goal = args.samples
            bar_w = int( (w-40) * min(1.0, N/float(goal)) )
            cv2.rectangle(disp, (20, h-30), (20+bar_w, h-12), (0,200,0), -1)
            cv2.rectangle(disp, (20, h-30), (w-20, h-12), (255,255,255), 2)
            status = "CAPTURING" if capturing else "IDLE"
            msg_top = f"{status} | Samples: {N}/{goal} | Rotate head CLOCKWISE, vary tilt/expressions. With & without glasses."
            cv2.putText(disp, msg_top, (12, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255,255,255), 2)

            if not face_present and capturing:
                cv2.putText(disp, "No face detected. Step closer / center face.", (12, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0,0,255), 2)

            # Auto-finish when goal reached
            if capturing and N >= goal:
                capturing = False
                # Save immediately
                arr = np.vstack(collected) if collected else np.zeros((0,128), dtype="float64")
                np.save(str(out_npy), arr)
                print(f"[SAVED] {out_npy} with shape {arr.shape}")
                # Update labels + roster
                labels = load_labels(labels_p)
                labels[person_name] = f"{person_name}"
                save_labels(labels_p, labels)
                upsert_roster(roster_p, person_id, person_name, card_uid)
                print(f"[UPDATED] labels.json and roster.csv")
                cv2.putText(disp, "Saved! You can quit (q) or press 'c' to capture again.", (12, 72), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,200,0), 2)

            # Show
            cv2.imshow("Auto Enroll (RealSense)", disp)
            key = cv2.waitKey(1) & 0xFF

            if key in (ord('q'), 27):
                break
            elif key == ord('c'):
                # restart capture
                collected.clear()
                last_enc = None
                last_sample_ts = 0.0
                capturing = True
                print("[CAPTURE] Started. Rotate head CLOCKWISE; include with/without glasses.")
            elif key == ord('r'):
                collected.clear()
                last_enc = None
                last_sample_ts = 0.0
                print("[RESET] Cleared collected samples.")
            elif key == ord('s'):
                # save early
                arr = np.vstack(collected) if collected else np.zeros((0,128), dtype="float64")
                np.save(str(out_npy), arr.astype("float32"))
                labels = load_labels(labels_p)
                labels[person_name] = f"{person_name}.npy"
                save_labels(labels_p, labels)
                upsert_roster(roster_p, person_id, person_name, card_uid)
                print(f"[SAVED EARLY] {out_npy} shape {arr.shape}; labels/roster updated.")
            elif key == ord('p'):
                try:
                    out = "/home/alpha/realsense_enroll.png"
                    cv2.imwrite(out, disp)
                    print(f"[📸] Saved {out}")
                except Exception as e:
                    print("[WARN] Snapshot failed:", e)

    except KeyboardInterrupt:
        pass
    finally:
        try:
            pipeline.stop()
        except Exception:
            pass
        cv2.destroyAllWindows()
        print("[INFO] Stopped.")

if __name__ == "__main__":
    main()
