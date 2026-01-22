import cv2
import dlib
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

PREDICTOR_PATH = os.path.join(
    BASE_DIR, "..", "models", "shape_predictor_68_face_landmarks.dat"
)

FACE_RECOGNITION_MODEL_PATH = os.path.join(
    BASE_DIR, "..", "models", "dlib_face_recognition_resnet_model_v1.dat"
)

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(PREDICTOR_PATH)
face_rec_model = dlib.face_recognition_model_v1(FACE_RECOGNITION_MODEL_PATH)

def main():
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("❌ Kamera açılamadı")
        return

    print("🎥 Kamera açıldı (Aşama 3 – dlib face recognition)")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = detector(gray)

        for face in faces:
            x1, y1 = face.left(), face.top()
            x2, y2 = face.right(), face.bottom()

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            shape = predictor(gray, face)
            face_descriptor = face_rec_model.compute_face_descriptor(frame, shape)

            # Şimdilik sadece yüzü algılıyoruz (tanıma Aşama 4)
        
        cv2.imshow("TraceFace AI - Stage 3", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
