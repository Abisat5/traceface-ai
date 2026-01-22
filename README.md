# TraceFace-AI

## 🇹🇷 Proje Açıklaması (TR)

**TraceFace with AI**, gerçek zamanlı yüz algılama ve takibi üzerine kurulu bir bilgisayarlı görü projesidir.  
Bu projede, bilgisayar kamerası kullanılarak canlı görüntü üzerinden yüzler tespit edilmekte ve algılanan yüzler çerçeve içine alınarak görsel olarak işaretlenmektedir.

### 🎯 Projenin Temel Amacı
- Yapay zeka destekli yüz algılama sistemlerinin temellerini öğrenmek
- Gerçek zamanlı kamera işleme mantığını kavramak
- Daha ileri aşamalarda (**yüz tanıma, kayıt, loglama, anormallik tespiti** vb.) geliştirilebilecek bir altyapı oluşturmaktır

---

### ✅ Şu Ana Kadar Yapılanlar (Aşama 1–2)
- Python sanal ortamı (**venv**) oluşturuldu
- Gerekli kütüphaneler kuruldu
- Bilgisayar kamerası başarıyla açıldı
- **OpenCV** ile gerçek zamanlı video akışı alındı
- **Haar Cascade** modeli kullanılarak yüz algılama yapıldı
- Algılanan yüzler **bounding box (dikdörtgen)** ile çizdirildi

---

### ⚙️ Bu Aşamada Sistem Ne Yapıyor?
- Kamerayı açar
- Görüntüyü sürekli okur
- Yüzleri algılar
- Algılanan yüzleri anlık olarak ekranda gösterir

---

### 🛠️ Kullanılan Teknolojiler ve Kütüphaneler
- **Python 3.10.9**
- **OpenCV (cv2)** – Kamera erişimi ve görüntü işleme
- **NumPy** – Görüntü verileriyle matematiksel işlemler
- **Haar Cascade Classifier** – Yüz algılama modeli

---

### 📌 Projenin Mevcut Durumu
✔ Kamera çalışıyor  
✔ Yüz algılama başarılı  
✔ Gerçek zamanlı çizim aktif  

---

### 🚀 Bir Sonraki Aşamalar
- Yüz tanıma
- Kişi eşleştirme
- Kayıt alma
- Alarm / log sistemi

---

## 🇬🇧 Project Description (EN)

**TraceFace with AI** is a computer vision project focused on real-time face detection and tracking.  
The system uses a computer camera to detect human faces in a live video stream and visually marks detected faces with bounding boxes.

### 🎯 Project Goals
- Learning the fundamentals of AI-based face detection systems
- Understanding real-time camera processing
- Building a solid foundation for future features such as **face recognition, logging, and anomaly detection**

---

### ✅ What Has Been Done So Far (Stage 1–2)
- Python virtual environment (**venv**) created
- Required dependencies installed
- Camera successfully accessed
- Real-time video stream captured using **OpenCV**
- Face detection implemented using **Haar Cascade**
- Detected faces are drawn with **bounding boxes**

---

### ⚙️ Current System Capabilities
- Opens the camera
- Continuously reads frames
- Detects faces
- Displays detected faces in real time

---

### 🛠️ Technologies and Libraries Used
- **Python 3.10.9**
- **OpenCV (cv2)** – Camera access and image processing
- **NumPy** – Mathematical operations on image data
- **Haar Cascade Classifier** – Face detection model

---

### 📌 Current Project Status
✔ Camera access working  
✔ Face detection successful  
✔ Real-time drawing enabled  

---

### 🚀 Next Stages
- Face recognition
- Identity matching
- Face registration
- Logging and alert systems
