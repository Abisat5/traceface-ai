import cv2
import face_recognition
import os
import numpy as np
import time

# ================= CONFIG =================
KNOWN_FACES_DIR = r"C:\Projects\traceface-ai\known_faces"
IMAGE_PATH = r"C:\Projects\traceface-ai\test_image.jpg"
WINDOW_NAME = "Face Recognition Viewer"

# ==========================================

# Load known faces
known_encodings = []
known_names = []

for file in os.listdir(KNOWN_FACES_DIR):
    if file.lower().endswith((".jpg", ".png", ".jpeg")):
        img = face_recognition.load_image_file(
            os.path.join(KNOWN_FACES_DIR, file)
        )
        enc = face_recognition.face_encodings(img)
        if enc:
            known_encodings.append(enc[0])
            known_names.append(os.path.splitext(file)[0])
            print(f"✅ Yüklendi: {file}")

# Load image
original = cv2.imread(IMAGE_PATH)
h, w = original.shape[:2]

scale = 1.0
offset_x, offset_y = 0, 0
dragging = False
last_x, last_y = 0, 0
last_action_time = 0

# ==========================================

def recognize_on_view(view):
    rgb = cv2.cvtColor(view, cv2.COLOR_BGR2RGB)
    locations = face_recognition.face_locations(rgb, model="hog")
    encodings = face_recognition.face_encodings(rgb, locations)

    for (top, right, bottom, left), face_enc in zip(locations, encodings):
        matches = face_recognition.compare_faces(
            known_encodings, face_enc, tolerance=0.5
        )
        name = "Unknown"

        if True in matches:
            idx = matches.index(True)
            name = known_names[idx]

        cv2.rectangle(view, (left, top), (right, bottom), (0, 255, 0), 2)
        cv2.putText(
            view,
            name,
            (left, top - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2
        )

    return view

# ==========================================

def mouse(event, x, y, flags, param):
    global dragging, last_x, last_y, offset_x, offset_y, scale, last_action_time

    if event == cv2.EVENT_LBUTTONDOWN:
        dragging = True
        last_x, last_y = x, y

    elif event == cv2.EVENT_MOUSEMOVE and dragging:
        dx = x - last_x
        dy = y - last_y
        offset_x -= dx
        offset_y -= dy
        last_x, last_y = x, y
        last_action_time = time.time()

    elif event == cv2.EVENT_LBUTTONUP:
        dragging = False

    elif event == cv2.EVENT_MOUSEWHEEL:
        if flags > 0:
            scale *= 1.1
        else:
            scale /= 1.1
        scale = max(0.3, min(scale, 3.0))
        last_action_time = time.time()

# ==========================================

cv2.namedWindow(WINDOW_NAME)
cv2.setMouseCallback(WINDOW_NAME, mouse)

while True:
    view_w = int(w / scale)
    view_h = int(h / scale)

    offset_x = max(0, min(offset_x, w - view_w))
    offset_y = max(0, min(offset_y, h - view_h))

    crop = original[
        offset_y:offset_y + view_h,
        offset_x:offset_x + view_w
    ]

    view = cv2.resize(crop, (w, h))

    # 🔥 SADECE HAREKETTEN SONRA TANIMA
    if time.time() - last_action_time < 0.2:
        view = recognize_on_view(view)

    cv2.imshow(WINDOW_NAME, view)

    if cv2.waitKey(20) & 0xFF == 27:
        break

cv2.destroyAllWindows()
