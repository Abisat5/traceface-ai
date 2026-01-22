import cv2
import face_recognition
import numpy as np
import os

# ==============================
# CONFIG
# ==============================
KNOWN_FACES_DIR = r"C:\Projects\traceface-ai\known_faces"
IMAGE_PATH = r"C:\Projects\traceface-ai\test_image.jpg"

# ==============================
# LOAD KNOWN FACES
# ==============================
known_encodings = []
known_names = []

for file in os.listdir(KNOWN_FACES_DIR):
    if file.lower().endswith((".jpg", ".png", ".jpeg")):
        img_path = os.path.join(KNOWN_FACES_DIR, file)
        image = face_recognition.load_image_file(img_path)
        encodings = face_recognition.face_encodings(image)

        if encodings:
            known_encodings.append(encodings[0])
            known_names.append(os.path.splitext(file)[0])
            print(f"✅ Yüklendi: {file}")
        else:
            print(f"⚠️ Yüz bulunamadı: {file}")

# ==============================
# LOAD IMAGE
# ==============================
original = cv2.imread(IMAGE_PATH)
if original is None:
    raise Exception("❌ Fotoğraf açılamadı")

image = original.copy()

# ==============================
# FACE RECOGNITION
# ==============================
rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
locations = face_recognition.face_locations(rgb)
encodings = face_recognition.face_encodings(rgb, locations)

for (top, right, bottom, left), face_encoding in zip(locations, encodings):
    matches = face_recognition.compare_faces(known_encodings, face_encoding, tolerance=0.5)
    name = "Unknown"

    if True in matches:
        name = known_names[matches.index(True)]

    cv2.rectangle(image, (left, top), (right, bottom), (0, 255, 0), 2)
    cv2.putText(image, name, (left, top - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

# ==============================
# PAN & ZOOM VARIABLES
# ==============================
zoom = 1.0
ox, oy = 0, 0
dragging = False
px, py = 0, 0

# ==============================
# MOUSE CALLBACK
# ==============================
def mouse(event, x, y, flags, param):
    global ox, oy, dragging, px, py, zoom

    if event == cv2.EVENT_LBUTTONDOWN:
        dragging = True
        px, py = x, y

    elif event == cv2.EVENT_MOUSEMOVE and dragging:
        ox += x - px
        oy += y - py
        px, py = x, y

    elif event == cv2.EVENT_LBUTTONUP:
        dragging = False

    elif event == cv2.EVENT_MOUSEWHEEL:
        if flags > 0:
            zoom *= 1.1
        else:
            zoom /= 1.1

# ==============================
# VIEWER LOOP
# ==============================
cv2.namedWindow("Face Viewer", cv2.WINDOW_NORMAL)
cv2.setMouseCallback("Face Viewer", mouse)

while True:
    h, w = image.shape[:2]
    resized = cv2.resize(image, (int(w * zoom), int(h * zoom)))

    canvas = np.zeros_like(original)
    y1 = max(0, oy)
    y2 = min(canvas.shape[0], oy + resized.shape[0])
    x1 = max(0, ox)
    x2 = min(canvas.shape[1], ox + resized.shape[1])

    ry1 = max(0, -oy)
    ry2 = ry1 + (y2 - y1)
    rx1 = max(0, -ox)
    rx2 = rx1 + (x2 - x1)

    canvas[y1:y2, x1:x2] = resized[ry1:ry2, rx1:rx2]

    cv2.imshow("Face Viewer", canvas)

    if cv2.waitKey(1) & 0xFF == 27:  # ESC
        break

cv2.destroyAllWindows()
