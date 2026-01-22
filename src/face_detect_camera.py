import cv2
import dlib
import time
import os


def main():
    # Kamera başlat
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("❌ Kamera açılamadı")
        return

    print("🎥 Kamera açıldı")
    print("➡️  Çıkış: q | Screenshot: s")

    # Face detector
    detector = dlib.get_frontal_face_detector()

    # FPS için
    prev_time = 0

    # Screenshot klasörü
    os.makedirs("captures", exist_ok=True)

    while True:
        ret, frame = cap.read()
        if not ret:
            print("❌ Kameradan görüntü alınamadı")
            break

        # FPS hesapla
        current_time = time.time()
        fps = 1 / (current_time - prev_time) if prev_time != 0 else 0
        prev_time = current_time

        # Griye çevir
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Yüzleri tespit et
        faces = detector(gray)

        # Yüz çizimleri
        for i, face in enumerate(faces):
            x1 = face.left()
            y1 = face.top()
            x2 = face.right()
            y2 = face.bottom()

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            cv2.putText(
                frame,
                f"Face #{i + 1}",
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2
            )

        # FPS yazısı
        cv2.putText(
            frame,
            f"FPS: {int(fps)}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2
        )

        # Ekrana bas
        cv2.imshow("TraceFace AI - Face Detection", frame)

        key = cv2.waitKey(1) & 0xFF

        # Screenshot
        if key == ord("s"):
            filename = f"captures/capture_{int(time.time())}.jpg"
            cv2.imwrite(filename, frame)
            print(f"📸 Screenshot alındı: {filename}")

        # Çıkış
        if key == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
