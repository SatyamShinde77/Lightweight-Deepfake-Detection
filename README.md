# Lightweight Deepfake Detection

# 🧠 Deepfake Detection System (Image & Video)

A deep learning–based system to detect deepfake content in **images and videos** using facial artifact analysis.

---

## 🚀 Features
- Image-based deepfake detection
- Video deepfake detection via multi-frame face analysis
- EfficientNet-B0 with transfer learning
- Face detection + face cropping for videos
- Confidence score output
- Interactive Gradio web interface

---

## 🛠️ Tech Stack
- Python
- PyTorch
- PyTorch Lightning
- EfficientNet-B0
- OpenCV
- Gradio

---

## 🧠 How It Works
1. **Images** are classified directly.
2. **Videos** are processed by:
   - Sampling multiple frames
   - Detecting and cropping faces
   - Classifying each face
   - Aggregating predictions for final verdict

This approach aligns inference with training data distribution and improves robustness.

---
## 📊 Model Evaluation Metrics

![Evaluation Results](https://raw.githubusercontent.com/SatyamShinde77/Lightweight-Deepfake-Detection/main/All%20metrics%20evaluation%20ss%20.png)

## 🎥 Demo Video
<video src="DeepFake checking Short video ....mp4" controls width="600"></video>



## ▶️ Run the Demo

```bash
pip install -r requirements.txt
python web-app.py
