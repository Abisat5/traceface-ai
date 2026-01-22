import cv2
import face_recognition
import os
import numpy as np
import time

KNOWN_FACES_DIR = r"C:\Projects\traceface-ai\known_faces"
IMAGE_PATH = r"C:\Projects\traceface-ai\test_image.jpg"
WINDOW = "Fast Face Viewer"

# ================= LOAD KNOWN FACES =================
known_encodings = []
known_names = []

for f in os.listdir(KNOWN_FACES_DIR):
    if f.lower().endswith((".jpg", ".png", ".jpeg")):
        img = face_recognition.load_image_file(os.path.join(KNOWN_FACES_DIR, f))
        enc = face_recognition.face_encodings(img)
        if enc:
            known_encodings.append(enc[0])
            known_names.append(os.path.splitext(f)[0])

# ================= IMAGE =================
original = cv2.imread(IMAGE_PATH)
H, W = original.shape[:2]

scale = 1.0
ox, oy = 0, 0
drag = False
lx, ly = 0, 0

last_recognition = 0
REC_INTERVAL = 0.3
cached_faces = []

# ================= FACE RECOGNITION =================
def recognize_fast(frame):
    small = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
    rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)

    locations = face_recognition.face_locations(rgb, model="hog")
    encodings = face_recognition.face_encodings(rgb, locations)

    results = []

    for (t, r, b, l), enc in zip(locations, encodings):
        matches = face_recognition.compare_faces(
            known_encodings, enc, tolerance=0.5
        )
        name = "Unknown"
        if True in matches:
            name = known_names[matches.index(True)]

        # scale back
        results.append((l*4, t*4, r*4, b*4, name))

    return results

# ================= MOUSE =================
def mouse(event, x, y, flags, param):
    global drag, lx, ly, ox, oy, scale

    if event == cv2.EVENT_LBUTTONDOWN:
        drag = True
        lx, ly = x, y

    elif event == cv2.EVENT_MOUSEMOVE and drag:
        ox -= (x - lx)
        oy -= (y - ly)
        lx, ly = x, y

    elif event == cv2.EVENT_LBUTTONUP:
        drag = False

    elif event == cv2.EVENT_MOUSEWHEEL:
        scale *= 1.1 if flags > 0 else 0.9
        scale = max(0.4, min(scale, 3.0))

# ================= LOOP =================
cv2.namedWindow(WINDOW)
cv2.setMouseCallback(WINDOW, mouse)

while True:
    vw, vh = int(W / scale), int(H / scale)
    ox = max(0, min(ox, W - vw))
    oy = max(0, min(oy, H - vh))

    crop = original[oy:oy+vh, ox:ox+vw]
    view = cv2.resize(crop, (W, H))

    now = time.time()
    if now - last_recognition > REC_INTERVAL:
        cached_faces = recognize_fast(view)
        last_recognition = now

    for l, t, r, b, name in cached_faces:
        cv2.rectangle(view, (l, t), (r, b), (0, 255, 0), 2)
        cv2.putText(view, name, (l, t-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)

    cv2.imshow(WINDOW, view)
    if cv2.waitKey(16) & 0xFF == 27:
        break

cv2.destroyAllWindows()
