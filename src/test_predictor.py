import dlib

predictor = dlib.shape_predictor(
    r"C:\Projects\traceface-ai\models\shape_predictor_68_face_landmarks.dat"
)

print("✅ Predictor başarıyla yüklendi")
