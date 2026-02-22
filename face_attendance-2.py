"""
FACE RECOGNITION ATTENDANCE SYSTEM (DeepFace)
===============================================
SETUP (run once in terminal):
    pip install deepface opencv-python numpy tf-keras

HOW TO USE:
    1. Run:  python face_attendance.py
    2. Press R → type name in terminal → look at camera → press SPACE
    3. Press A → look at camera → attendance prints in terminal
    4. Press Q → quit and see final report
"""

import cv2
import os
import numpy as np
import pickle
import warnings
from datetime import datetime

warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

try:
    from deepface import DeepFace
except ImportError:
    print("\n❌  Run this first:")
    print("    pip install deepface opencv-python numpy tf-keras\n")
    exit(1)

# ── Settings ─────────────────────────────────────────────────────────────────
FACES_FILE = "known_faces.pkl"   # saved face embeddings
MODEL      = "Facenet"           # fast + accurate
THRESHOLD  = 10                  # distance threshold (lower = stricter)

# ── Load saved faces ──────────────────────────────────────────────────────────
known_embeddings = []
known_names      = []

if os.path.exists(FACES_FILE):
    with open(FACES_FILE, "rb") as f:
        data = pickle.load(f)
    known_embeddings = data["embeddings"]
    known_names      = data["names"]
    print(f"\n✅  Loaded {len(set(known_names))} face(s): {', '.join(set(known_names))}")
else:
    print("\nℹ️   No faces registered yet. Press R to register.")

# ── Attendance log ────────────────────────────────────────────────────────────
attendance = {}   # name → time string


def get_embedding(face_img):
    """Get face embedding from a cropped face image."""
    try:
        result = DeepFace.represent(
            img_path      = face_img,
            model_name    = MODEL,
            enforce_detection = False
        )
        return np.array(result[0]["embedding"])
    except Exception:
        return None


def find_match(embedding):
    """Compare embedding against all known faces. Returns (name, distance)."""
    if not known_embeddings:
        return None, None

    distances = [
        np.linalg.norm(embedding - np.array(e))
        for e in known_embeddings
    ]
    best_idx  = int(np.argmin(distances))
    best_dist = distances[best_idx]

    if best_dist < THRESHOLD:
        return known_names[best_idx], round(best_dist, 2)
    return None, best_dist


# ── Camera setup ──────────────────────────────────────────────────────────────
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("❌  Cannot open webcam.")
    exit(1)

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

mode         = "IDLE"
pending_name = ""
flash        = 0
flash_msg    = ""
flash_ok     = True

print("\n📷  Camera ready!")
print("    R = Register   A = Attendance   SPACE = Capture   Q = Quit\n")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame  = cv2.flip(frame, 1)
    h, w   = frame.shape[:2]
    gray   = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces using OpenCV (fast)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60))

    for (x, y, fw, fh) in faces:
        face_crop = frame[y:y+fh, x:x+fw]
        name      = "Unknown"
        color     = (50, 50, 255)   # red

        # Only run DeepFace in ATTENDANCE mode to keep it fast
        if mode == "ATTENDANCE" and len(face_crop) > 0:
            emb = get_embedding(face_crop)
            if emb is not None:
                matched, dist = find_match(emb)
                if matched:
                    name  = matched
                    color = (0, 220, 80)   # green

                    if name not in attendance:
                        attendance[name] = datetime.now().strftime("%H:%M:%S")
                        print(f"  ✅  PRESENT → {name}  ({attendance[name]})")
                        flash_msg = f"{name} marked present!"
                        flash_ok  = True
                        flash     = 80

        elif mode == "REGISTER":
            color = (0, 200, 255)   # yellow during register

        # Draw box + label
        cv2.rectangle(frame, (x, y), (x+fw, y+fh), color, 2)
        label = name + (" ✓" if name in attendance else "")
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_DUPLEX, 0.6, 1)
        cv2.rectangle(frame, (x, y+fh), (x+tw+10, y+fh+26), color, -1)
        cv2.putText(frame, label, (x+5, y+fh+18),
                    cv2.FONT_HERSHEY_DUPLEX, 0.6, (0,0,0), 1, cv2.LINE_AA)

        if mode == "REGISTER" and pending_name:
            cv2.putText(frame, "Press SPACE to capture", (x, y-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0,200,255), 1, cv2.LINE_AA)

    # ── HUD top bar ───────────────────────────────────────────────────────
    cv2.rectangle(frame, (0,0), (w,46), (18,18,18), -1)
    mode_color = (0,220,80) if mode=="ATTENDANCE" else (0,200,255) if mode=="REGISTER" else (160,160,160)
    cv2.putText(frame, f"MODE: {mode}", (12,30),
                cv2.FONT_HERSHEY_DUPLEX, 0.7, mode_color, 1, cv2.LINE_AA)
    cv2.putText(frame, f"Registered: {len(set(known_names))}   Present: {len(attendance)}", (w-310,30),
                cv2.FONT_HERSHEY_DUPLEX, 0.52, (200,200,200), 1, cv2.LINE_AA)

    # ── Register prompt ───────────────────────────────────────────────────
    if mode == "REGISTER":
        cv2.rectangle(frame, (0,46), (w,82), (30,30,30), -1)
        cv2.putText(frame, f"Registering: {pending_name}  |  SPACE = capture", (12,68),
                    cv2.FONT_HERSHEY_DUPLEX, 0.6, (0,200,255), 1, cv2.LINE_AA)

    # ── Bottom controls ───────────────────────────────────────────────────
    cv2.rectangle(frame, (0,h-34), (w,h), (18,18,18), -1)
    cv2.putText(frame, "[R] Register   [A] Attendance   [SPACE] Capture   [Q] Quit",
                (10, h-10), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (120,120,120), 1, cv2.LINE_AA)

    # ── Flash message ─────────────────────────────────────────────────────
    if flash > 0:
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, h//2-38), (w, h//2+38), (10,10,10), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        fc = (0,220,80) if flash_ok else (50,50,255)
        cx = max(10, w//2 - len(flash_msg)*9)
        cv2.putText(frame, flash_msg, (cx, h//2+12),
                    cv2.FONT_HERSHEY_DUPLEX, 0.85, fc, 2, cv2.LINE_AA)
        flash -= 1

    cv2.imshow("Attendance System", frame)

    # ── Key handling ──────────────────────────────────────────────────────
    key = cv2.waitKey(1) & 0xFF

    if key in (ord('q'), ord('Q'), 27):
        break

    elif key in (ord('r'), ord('R')):
        cv2.destroyAllWindows()
        name_input = input("\n📝  Type name then press ENTER: ").strip()
        if name_input:
            pending_name = name_input
            mode         = "REGISTER"
            print(f"    Look at the camera and press SPACE to capture '{pending_name}'.")
        else:
            print("    Cancelled.")
            mode = "IDLE"

    elif key in (ord('a'), ord('A')):
        mode = "ATTENDANCE"
        print("\n🎯  ATTENDANCE MODE — look at the camera\n")

    elif key == ord(' ') and mode == "REGISTER" and pending_name:
        # Capture and save face
        small      = cv2.resize(frame, (0,0), fx=0.5, fy=0.5)
        gray2      = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        faces2     = face_cascade.detectMultiScale(gray2, 1.1, 5, minSize=(40,40))

        if len(faces2) > 0:
            (x2,y2,fw2,fh2) = faces2[0]
            x2,y2,fw2,fh2   = x2*2, y2*2, fw2*2, fh2*2
            face_crop2       = frame[y2:y2+fh2, x2:x2+fw2]
            emb              = get_embedding(face_crop2)

            if emb is not None:
                known_embeddings.append(emb.tolist())
                known_names.append(pending_name)
                with open(FACES_FILE, "wb") as f:
                    pickle.dump({"embeddings": known_embeddings, "names": known_names}, f)
                print(f"  ✅  Registered: {pending_name}")
                flash_msg    = f"Registered: {pending_name}!"
                flash_ok     = True
                flash        = 80
                mode         = "IDLE"
                pending_name = ""
            else:
                print("  ⚠️   Could not extract face features. Try again.")
                flash_msg = "Try again — face unclear!"
                flash_ok  = False
                flash     = 60
        else:
            print("  ⚠️   No face detected. Make sure your face is visible.")
            flash_msg = "No face detected!"
            flash_ok  = False
            flash     = 60

cap.release()
cv2.destroyAllWindows()

# ── Final Report ──────────────────────────────────────────────────────────────
print("\n" + "="*45)
print("         ATTENDANCE REPORT")
print("="*45)
if attendance:
    for i, (name, time) in enumerate(attendance.items(), 1):
        print(f"  {i}.  {name:<22} {time}")
    print(f"\n  Total present : {len(attendance)}")
    print(f"  Absent        : {max(0, len(set(known_names)) - len(attendance))}")
else:
    print("  No attendance recorded this session.")
print("="*45 + "\n")
