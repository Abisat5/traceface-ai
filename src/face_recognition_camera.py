import cv2
import dlib
import os
import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

PREDICTOR_PATH = os.path.join(BASE_DIR, "..", "models", "shape_predictor_68_face_landmarks.dat")
FACE_RECOG_MODEL_PATH = os.path.join(BASE_DIR, "..", "models", "dlib_face_recognition_resnet_model_v1.dat")
KNOWN_FACES_DIR = os.path.join(BASE_DIR, "..", "known_faces")

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(PREDICTOR_PATH)
face_rec_model = dlib.face_recognition_model_v1(FACE_RECOG_MODEL_PATH)

def load_known_faces():
    known_encodings = []
    known_names = []

    for file in os.listdir(KNOWN_FACES_DIR):
        if file.lower().endswith((".jpg", ".png", ".jpeg")):
            name = os.path.splitext(file)[0]

            img_path = os.path.join(KNOWN_FACES_DIR, file)
            img = cv2.imread(img_path)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            faces = detector(gray)
            if len(faces) == 0:
                continue

            shape = predictor(gray, faces[0])
            encoding = face_rec_model.compute_face_descriptor(img, shape)

            known_encodings.append(np.array(encoding))
            known_names.append(name)

    return known_encodings, known_names

def main():
    known_encodings, known_names = load_known_faces()

    cap = cv2.VideoCapture(0)
    print("🎥 Kamera açıldı – Aşama 4")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = detector(gray)

        for face in faces:
            x1, y1 = face.left(), face.top()
            x2, y2 = face.right(), face.bottom()

            shape = predictor(gray, face)
            face_encoding = np.array(
                face_rec_model.compute_face_descriptor(frame, shape)
            )

            name = "UNKNOWN"

            if known_encodings:
                distances = np.linalg.norm(known_encodings - face_encoding, axis=1)
                min_dist = np.min(distances)

                if min_dist < 0.6:
                    name = known_names[np.argmin(distances)]

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(
                frame,
                name,
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (0, 255, 0),
                2
            )

        cv2.imshow("TraceFace AI - Stage 4", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
