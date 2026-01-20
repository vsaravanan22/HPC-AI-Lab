#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import os, sys, json, csv, time, argparse, datetime as dt, tempfile, subprocess, shlex
from pathlib import Path
from typing import Dict, List, Tuple
from smbus2 import SMBus, i2c_msg
import adafruit_mlx90640

import numpy as np
import cv2
import pyrealsense2 as rs
import face_recognition

# --- NFC / PN532 imports (same stack as read_card.py) ---
import board
import busio
from adafruit_pn532.i2c import PN532_I2C

# ---------------- Defaults ----------------
DETECT_PROTO = "/home/alpha/face_models/deploy.prototxt"
DETECT_MODEL = "/home/alpha/face_models/res10_300x300_ssd_iter_140000.caffemodel"
EMBED_DIR    = "/home/alpha/embeddings"
LABELS_PATH  = "/home/alpha/embeddings/labels.json"
ROSTER_PATH  = "/home/alpha/roster.csv"
ATTEND_CSV   = "/home/alpha/attendance.csv"

# TTS tuning
TTS_PITCH_CENTS  = -250
TTS_TEMPO_FACTOR = 1.25
TTS_GAIN_DB      = -1.5

TEMP_ALERT_F = 100.0  # Temperature alert threshold in Fahrenheit

def today_key() -> str:
    return dt.date.today().isoformat()


def ensure_parent(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)


def load_labels(labels_path: Path) -> Dict[str, str]:
    if labels_path.exists():
        try:
            with labels_path.open("r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, dict):
                return {str(k): str(v) for k, v in data.items()}
        except Exception:
            pass
    return {}


def load_embeddings(emb_dir: Path, labels: Dict[str, str]) -> Tuple[List[np.ndarray], List[str]]:
    encs: List[np.ndarray] = []
    names: List[str] = []
    for npy in sorted(emb_dir.glob("*.npy")):
        try:
            arr = np.load(str(npy))
            if arr.ndim == 1:
                arr = arr[None, :]
            if arr.shape[-1] != 128:
                print(f"[WARN] Skip {npy.name}: not 128D (shape {arr.shape})")
                continue
            disp = labels.get(npy.name, labels.get(npy.stem, npy.stem))
            for row in arr:
                encs.append(row.astype("float64"))
                names.append(disp)
        except Exception as e:
            print(f"[WARN] Failed to load {npy.name}: {e}")
    print(f"[INFO] Loaded {len(encs)} embeddings for {len(set(names))} identities.")
    return encs, names


def detect_faces(net, frame_bgr, conf_thresh=0.55):
    (h, w) = frame_bgr.shape[:2]
    blob = cv2.dnn.blobFromImage(
        frame_bgr,
        1.0,
        (300, 300),
        (104.0, 177.0, 123.0),
        swapRB=False,
        crop=False,
    )
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
            x1 = max(0, min(x1, w - 1))
            y1 = max(0, min(y1, h - 1))
            x2 = max(0, min(x2, w - 1))
            y2 = max(0, min(y2, h - 1))
            if x2 > x1 and y2 > y1:
                boxes.append((x1, y1, x2, y2, conf))
    return boxes


def expand_box(x1, y1, x2, y2, img_w, img_h, scale=0.25):
    w = x2 - x1
    h = y2 - y1
    cx = x1 + w // 2
    cy = y1 + h // 2
    new_w = int(w * (1.0 + scale))
    new_h = int(h * (1.0 + scale))
    nx1 = max(0, cx - new_w // 2)
    ny1 = max(0, cy - new_h // 2)
    nx2 = min(img_w - 1, cx + new_w // 2)
    ny2 = min(img_h - 1, cy + new_h // 2)
    return nx1, ny1, nx2, ny2


def read_roster_csv(roster_path: Path) -> Dict[str, Dict[str, str]]:
    m: Dict[str, Dict[str, str]] = {}
    if roster_path.exists():
        with roster_path.open("r", encoding="utf-8") as f:
            r = csv.DictReader(f)
            for row in r:
                nm = (row.get("name") or "").strip()
                pid = (row.get("id") or "").strip()
                cu = (row.get("card_uid") or "").strip()
                if nm:
                    m[nm] = {"id": pid, "card_uid": cu}
    return m


def append_attendance(csv_path: Path, name: str, conf: float):
    ensure_parent(csv_path)
    new_file = not csv_path.exists()
    with csv_path.open("a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if new_file:
            w.writerow(["date", "time", "name", "confidence", "action"])
        now = dt.datetime.now()
        w.writerow(
            [
                now.date().isoformat(),
                now.strftime("%H:%M:%S"),
                name,
                f"{conf:.3f}",
                "present",
            ]
        )


def already_marked_today(csv_path: Path, name: str) -> bool:
    if not csv_path.exists():
        return False
    t = today_key()
    try:
        with csv_path.open("r", encoding="utf-8") as f:
            r = csv.DictReader(f)
            for row in r:
                if (row.get("date") == t) and (row.get("name") or "").strip() == name:
                    return True
    except Exception:
        pass
    return False


def which(cmd: str) -> bool:
    from shutil import which as _w
    return _w(cmd) is not None


def speak(msg: str):
    if which("pico2wave") and which("sox") and which("paplay"):
        try:
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as src_wav:
                src_path = src_wav.name
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as out_wav:
                out_path = out_wav.name
            subprocess.run(
                ["pico2wave", "-w", src_path, msg],
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            sox_cmd = (
                f"sox {shlex.quote(src_path)} -b 16 {shlex.quote(out_path)} "
                f"pitch {TTS_PITCH_CENTS} tempo {TTS_TEMPO_FACTOR} gain -n {TTS_GAIN_DB}"
            )
            subprocess.run(
                sox_cmd,
                shell=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            subprocess.run(
                ["paplay", out_path],
                check=False,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        except Exception as e:
            print(f"[WARN] TTS failed (pico/sox): {e}")
        finally:
            try:
                os.remove(src_path)
            except Exception:
                pass
            try:
                os.remove(out_path)
            except Exception:
                pass
    else:
        print("[WARN] No TTS engine available.")

class CP_I2C_Adapter:
    """
    CircuitPython-style I2C object backed by smbus2 low-level i2c_rdwr.
    Implements the methods the Adafruit MLX90640 driver expects.
    """
    def __init__(self, busnum: int):
        self.bus = SMBus(busnum)

    def try_lock(self) -> bool:
        return True

    def unlock(self) -> None:
        pass

    def writeto(self, address: int, buffer: bytes, *, start: int = 0, end: int | None = None) -> None:
        if end is None:
            end = len(buffer)
        data = bytes(buffer[start:end])
        if len(data) == 0:
            return
        msg = i2c_msg.write(address, data)
        self.bus.i2c_rdwr(msg)

    def readfrom_into(self, address: int, buffer: bytearray, *, start: int = 0, end: int | None = None) -> None:
        if end is None:
            end = len(buffer)
        length = end - start
        if length <= 0:
            return
        msg = i2c_msg.read(address, length)
        self.bus.i2c_rdwr(msg)
        buffer[start:end] = bytes(msg)

    def writeto_then_readfrom(
        self,
        address: int,
        out_buffer: bytes,
        in_buffer: bytearray,
        *,
        out_start: int = 0,
        out_end: int | None = None,
        in_start: int = 0,
        in_end: int | None = None,
    ) -> None:
        if out_end is None:
            out_end = len(out_buffer)
        if in_end is None:
            in_end = len(in_buffer)

        out_data = bytes(out_buffer[out_start:out_end])
        in_length = in_end - in_start

        msgs = []
        if len(out_data) > 0:
            msgs.append(i2c_msg.write(address, out_data))
        if in_length > 0:
            msgs.append(i2c_msg.read(address, in_length))

        if not msgs:
            return

        self.bus.i2c_rdwr(*msgs)
        if in_length > 0:
            in_buffer[in_start:in_end] = bytes(msgs[-1])

    def close(self) -> None:
        self.bus.close()

def init_pn532():
    """
    Initialize PN532 via I2C using Adafruit Blinka stack.
    Returns pn532 object or None if failed.
    """
    try:
        i2c = busio.I2C(board.SCL, board.SDA)
    except Exception as e:
        print(f"[PN532] ERROR: Could not initialize I2C: {e}")
        return None

    try:
        pn532 = PN532_I2C(i2c, debug=False)
    except Exception as e:
        print(f"[PN532] ERROR: Could not initialize PN532: {e}")
        return None

    try:
        ic, ver, rev, support = pn532.firmware_version
        print(f"[PN532] Found PN532 with firmware {ver}.{rev}")
    except Exception as e:
        print(f"[PN532] ERROR: Could not read firmware: {e}")
        return None

    pn532.SAM_configuration()
    print("[PN532] Ready for card scans.")
    return pn532


def main():
    ap = argparse.ArgumentParser(
        description="Attendance w/ RealSense + NFC card verify, once-per-day, speech."
    )
    ap.add_argument("--det-prototxt", default=DETECT_PROTO)
    ap.add_argument("--det-caffemodel", default=DETECT_MODEL)
    ap.add_argument("--embeddings", default=EMBED_DIR)
    ap.add_argument("--labels", default=LABELS_PATH)
    ap.add_argument("--roster", default=ROSTER_PATH)
    ap.add_argument("--attendance", default=ATTEND_CSV)
    ap.add_argument("--tolerance", type=float, default=0.50)
    ap.add_argument("--det-conf", type=float, default=0.55)
    ap.add_argument("--announce-cooldown", type=float, default=8.0)
    ap.add_argument("--depth", action="store_true")
    ap.add_argument("--speak-test", action="store_true")
    args = ap.parse_args()

    if args.speak_test:
        speak("This is a system check.")
        print("[SPEAK] TTS test sent.")
        return

    det_proto = Path(args.det_prototxt)
    det_model = Path(args.det_caffemodel)
    emb_dir = Path(args.embeddings)
    labels_p = Path(args.labels)
    roster_p = Path(args.roster)
    csv_path = Path(args.attendance)

    labels = load_labels(labels_p)
    known_encs, known_names = load_embeddings(emb_dir, labels)
    roster_map = read_roster_csv(roster_p)  # name -> {"id", "card_uid"}

    if not det_proto.exists() or not det_model.exists():
        print("[ERROR] Face detector model files not found.")
        sys.exit(1)

    net = cv2.dnn.readNetFromCaffe(str(det_proto), str(det_model))

    # --- Initialize PN532 (NFC) ---
    pn532 = init_pn532()
    if pn532 is None:
        print("[WARN] NFC card verification DISABLED (PN532 not available).")

    # --- MLX90640 (Thermal) setup ---
    mlx = None
    mlx_i2c = None
    thermal_frame = None
    try:
        mlx_i2c = CP_I2C_Adapter(1)  # MLX is on /dev/i2c-1
        mlx = adafruit_mlx90640.MLX90640(mlx_i2c)
        mlx.refresh_rate = adafruit_mlx90640.RefreshRate.REFRESH_4_HZ
        thermal_frame = np.zeros(32 * 24, dtype=float)
        # throw away a few startup frames to avoid spiky first read
        for _ in range(5):
            mlx.getFrame(thermal_frame)
        print("[THERMAL] MLX90640 initialized.")
    except Exception as e:
        print(f"[THERMAL] Disabled (MLX90640 init failed): {e}")
        mlx = None

    # --- RealSense setup ---
    pipeline = rs.pipeline()
    cfg = rs.config()
    cfg.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    if args.depth:
        cfg.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    profile = pipeline.start(cfg)
    align = rs.align(rs.stream.color) if args.depth else None
    colorizer = rs.colorizer() if args.depth else None

    last_announce: Dict[str, float] = {}
    present_set_today = set()

    # For NFC-based verification
    pending_card = {"name": None, "expires": 0.0}

    print("[INFO] Running. Press 'q' to quit, 'p' to snapshot.")
    try:
        while True:
            frames = pipeline.wait_for_frames()
            if args.depth:
                frames = align.process(frames)
                depth_frame = frames.get_depth_frame()
            else:
                depth_frame = None
            color_frame = frames.get_color_frame()
            if not color_frame:
                continue

            frame = np.asanyarray(color_frame.get_data())
            disp = frame.copy()
            h, w = frame.shape[:2]

            boxes = detect_faces(net, frame, conf_thresh=args.det_conf)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            face_results = []
            for (x1, y1, x2, y2, conf) in boxes:
                ex1, ey1, ex2, ey2 = expand_box(x1, y1, x2, y2, w, h, scale=0.25)
                loc = [(ey1, ex2, ey2, ex1)]
                encs = face_recognition.face_encodings(
                    rgb, known_face_locations=loc, num_jitters=1
                )
                name = "Unknown"
                score = 0.0
                if len(encs) > 0 and len(known_encs) > 0:
                    cand = encs[0]
                    dists = face_recognition.face_distance(known_encs, cand)
                    idx = int(np.argmin(dists))
                    best = float(dists[idx])
                    if best <= args.tolerance:
                        name = known_names[idx]
                        score = max(0.0, 1.0 - (best / max(args.tolerance, 1e-6)))
                    else:
                        score = max(0.0, 1.0 - (best / 0.6))
                face_results.append(((ex1, ey1, ex2, ey2), name, score))

            # Draw faces & handle NFC prompts
            for (ex1, ey1, ex2, ey2), name, score in face_results:
                color = (0, 200, 0) if name != "Unknown" else (0, 0, 255)
                cv2.rectangle(disp, (ex1, ey1), (ex2, ey2), color, 2)
                label = f"{name} {score*100:.1f}%"
                cv2.putText(
                    disp,
                    label,
                    (ex1, max(20, ey1 - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    color,
                    2,
                )

                if name != "Unknown" and score >= 0.48:
                    now = time.time()
                    last = last_announce.get(name, 0.0)
                    allow = (now - last) >= float(args.announce_cooldown)

                    if (
                        (today_key(), name) in present_set_today
                        or already_marked_today(csv_path, name)
                    ):
                        continue

                    # Only use NFC if pn532 available
                    if pn532 is None:
                        # Fallback: auto-mark without NFC
                        if not already_marked_today(csv_path, name):
                            append_attendance(csv_path, name, conf=score)
                            present_set_today.add((today_key(), name))
                        if allow:
                            speak(
                                f"Welcome to class, {name}. I hope you have a great day."
                            )
                            last_announce[name] = time.time()
                        continue

                    # Start / refresh pending NFC verification for this name
                    if (
                        pending_card["name"] is None
                        or pending_card["name"] != name
                        or now > pending_card["expires"]
                    ):
                        pending_card["name"] = name
                        pending_card["expires"] = now + 15.0  # 15s to scan card
                        if allow:
                            speak(f"{name}, please scan your card.")
                            last_announce[name] = time.time()

            # Overlay NFC prompt if waiting for a card
            now_t = time.time()
            if pending_card["name"] is not None and now_t < pending_card["expires"]:
                overlay = disp.copy()
                txt = f"Scan card for {pending_card['name']}"
                (tw, th), _ = cv2.getTextSize(
                    txt, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2
                )
                cv2.rectangle(
                    overlay,
                    (10, 10),
                    (10 + tw + 20, 10 + th + 30),
                    (0, 0, 0),
                    -1,
                )
                cv2.addWeighted(overlay, 0.5, disp, 0.5, 0, disp)
                cv2.putText(
                    disp,
                    txt,
                    (20, 20 + th),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (255, 255, 255),
                    2,
                )

                # Poll PN532 for a card while in this pending state
                if pn532 is not None:
                    uid = pn532.read_passive_target(timeout=0.1)
                    if uid is not None:
                        scanned_uid = "".join(f"{b:02X}" for b in uid)
                        name = pending_card["name"]
                        info = roster_map.get(name, {})
                        expected_uid = (info.get("card_uid") or "").upper()

                        print(
                            f"[NFC] Scanned UID {scanned_uid} for {name}, expected {expected_uid}"
                        )

                        # Require a matching card_uid; if roster has no card_uid,
                        # treat as mismatch (forces re-enrollment with card).
                        if expected_uid and scanned_uid.upper() == expected_uid:
                            if not already_marked_today(csv_path, name):
                                append_attendance(csv_path, name, conf=0.99)
                                present_set_today.add((today_key(), name))
                            speak(
                                f"Welcome to class, {name}. I hope you have a great day."
                            )
                            pending_card["name"] = None
                            pending_card["expires"] = 0.0
                        else:
                            speak("Card does not match. Please try again.")
                            # Extend the window a bit for another try
                            pending_card["expires"] = time.time() + 10.0

            # Build display (optionally with depth)
            if args.depth and depth_frame is not None:
                depth_color = np.asanyarray(
                    colorizer.colorize(depth_frame).get_data()
                )
                if depth_color.shape[:2] != disp.shape[:2]:
                    depth_color = cv2.resize(
                        depth_color,
                        (disp.shape[1], disp.shape[0]),
                        interpolation=cv2.INTER_NEAREST,
                    )
                combo = np.hstack([disp, depth_color])
            else:
                combo = disp
            
            # --- Thermal max overlay (F) ---
            if mlx is not None and thermal_frame is not None:
                try:
                    mlx.getFrame(thermal_frame)
                    max_c = float(np.max(thermal_frame))
                    max_f = (max_c * 9.0 / 5.0) + 32.0
                    cv2.putText(
                        combo,
                        f"Thermal Max: {max_f:.1f} F",
                        (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        (255, 255, 255),
                        2,
                    )

                            # 🔥 Temperature alert
                    if max_f >= TEMP_ALERT_F:
                        cv2.putText(
                            combo,
                            "⚠ TEMP ALERT ⚠",
                            (10, 65),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            1.0,
                            (0, 0, 255),  # Red alert
                            3,
            )
                except Exception:
                    pass

            cv2.imshow("Attendance (Color | Depth)", combo)
            key = cv2.waitKey(1) & 0xFF
            if key != 255:
                if key in (ord("q"), 27):
                    break
                elif key == ord("p"):
                    try:
                        out = "/home/alpha/realsense_snapshot.png"
                        cv2.imwrite(out, combo)
                        print(f"[📸] Saved {out}")
                    except Exception as e:
                        print(f"[WARN] Snapshot failed: {e}")
                # NOTE: no more ID entry via keyboard – card only.

    except KeyboardInterrupt:
        pass
    finally:
        try:
            pipeline.stop()
        except Exception:
            pass
        try:
            if mlx_i2c is not None:
                mlx_i2c.close()
        except Exception:
            pass
        cv2.destroyAllWindows()
        print("[INFO] Stopped.")


if __name__ == "__main__":
    main()