# TraceFace AI

## 🇹🇷 Türkçe

**TraceFace AI**, yapay zeka ve bilgisayarlı görü teknikleri kullanarak yüz algılama ve yüz tanıma işlemlerini gerçekleştiren bir projedir.  
Proje, hem **gerçek zamanlı kamera akışı** hem de **statik görüntüler** üzerinde çalışabilecek şekilde adım adım geliştirilmektedir.

Amaç, kayıp kişi / aranan şahıs tespiti gibi senaryolar için güçlü ve genişletilebilir bir altyapı oluşturmaktır.

---

## Projenin Amaçları

- Yapay zeka destekli yüz algılama ve tanıma sistemlerini öğrenmek
- Gerçek zamanlı ve görüntü tabanlı yüz işleme mantığını kavramak
- Performans optimizasyonu (FPS, cache, ölçekleme) uygulamalarını deneyimlemek
- Akademik ve profesyonel portföyde kullanılabilecek bir proje geliştirmek

---

## Şu Ana Kadar Yapılanlar

### Aşama 1 – Gerçek Zamanlı Yüz Algılama
- Python sanal ortamı (venv) oluşturuldu
- Gerekli kütüphaneler kuruldu
- OpenCV ile kamera erişimi sağlandı
- Haar Cascade modeli kullanılarak yüz algılama gerçekleştirildi
- Algılanan yüzler canlı görüntü üzerinde işaretlendi

### Aşama 2 – Altyapı ve Model Yapılandırması
- Proje klasör yapısı oluşturuldu
- Model dosyaları (`shape_predictor_68_face_landmarks.dat`) harici klasöre taşındı
- dlib ve face_recognition bağımlılıkları test edildi
- Model yükleme ve doğrulama testleri başarıyla tamamlandı

### Aşama 3 – Görüntü Üzerinde Yüz Tanıma (Image-Based Recognition)
- `known_faces/` klasörü üzerinden bilinen yüzler yüklendi
- Statik bir fotoğraf üzerinde yüz tanıma eklendi
- Mouse ile:
  - Görüntü sürükleme (pan)
  - Yakınlaştırma / uzaklaştırma (zoom)
- FPS artırımı için:
  - Görüntü ölçekleme
  - Aralıklı yüz tanıma
  - Tanıma sonuçlarının cache’lenmesi
- Kullanıcı görüntüyü her hareket ettirdiğinde yüz tanıma yeniden çalışacak şekilde optimize edildi

---

## Kullanılan Teknolojiler ve Kütüphaneler

- Python 3.10.9
- OpenCV (cv2) – Kamera ve görüntü işleme
- NumPy – Görüntü verisi işlemleri
- dlib – Yüz landmark ve tanıma altyapısı
- face_recognition – Yüksek seviye yüz tanıma API
- Haar Cascade Classifier – Yüz algılama modeli

---

## Proje Yapısı

traceface-ai/
│
├── src/
│ ├── face_detect_camera.py
│ ├── face_recognition_camera.py
│ ├── face_recognition_image.py
│ ├── image_face_recognition_fast.py
│ ├── image_face_recognition_pan_zoom.py
│ ├── image_face_recognition_viewer.py
│ ├── test_predictor.py
│
├── known_faces/
│ ├── person1/
│ ├── person2/
│
├── models/
│ └── shape_predictor_68_face_landmarks.dat
│
├── venv/
├── README.md
└── requirements.txt


---

## Mevcut Durum

- Kamera üzerinden yüz algılama çalışıyor
- Statik görüntülerde yüz tanıma çalışıyor
- Pan & zoom destekli etkileşimli görüntüleme mevcut
- Performans optimizasyonları uygulanmış durumda

---

## Planlanan Geliştirmeler

- Gerçek zamanlı yüz tanıma
- Kişi kayıt ve veri tabanı entegrasyonu
- Loglama ve zaman damgalı kayıt sistemi
- Alarm / bildirim mekanizması
- Web arayüz (dashboard)
- Video dosyası üzerinden analiz

---

## 🇬🇧 English

**TraceFace AI** is a computer vision project focused on face detection and face recognition using artificial intelligence techniques.  
The system is designed to work with both **real-time camera streams** and **static images**, following a step-by-step development approach.

The long-term goal is to build a scalable foundation for scenarios such as missing person or wanted individual detection systems.

---

## Project Goals

- Learn AI-based face detection and recognition systems
- Understand real-time and image-based vision processing
- Apply performance optimization techniques (FPS, caching, scaling)
- Build a strong portfolio project suitable for internships and professional use

---

## Progress So Far

### Stage 1 – Real-Time Face Detection
- Python virtual environment created
- Required dependencies installed
- Camera access via OpenCV
- Face detection using Haar Cascade
- Live bounding box drawing

### Stage 2 – Infrastructure & Model Setup
- Clean project structure established
- External model management implemented
- dlib and face_recognition validated
- Predictor loading tested successfully

### Stage 3 – Image-Based Face Recognition
- Known faces loaded from `known_faces/` directory
- Face recognition on static images implemented
- Interactive image viewer with:
  - Mouse-based pan
  - Zoom in / out
- Performance optimizations:
  - Image downscaling
  - Timed recognition
  - Recognition result caching
- Recognition recalculated dynamically during image navigation

---

## Technologies & Libraries

- Python 3.10.9
- OpenCV
- NumPy
- dlib
- face_recognition
- Haar Cascade Classifier

---

## Current Status

- Real-time face detection: working
- Image-based face recognition: working
- Interactive pan & zoom: implemented
- Optimized FPS and smooth user experience

---

## Upcoming Features

- Real-time face recognition
- Face registration and database integration
- Logging and alert system
- Web-based dashboard
- Video file analysis
