import gradio as gr
import torch
import mimetypes
import cv2
import os

from PIL import Image
from torchvision import transforms
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights

from lightning_modules.detector import DeepfakeDetector

# ===============================
# Face Detector (LIGHTWEIGHT)
# ===============================
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

# ===============================
# Load Model (CORRECT WAY)
# ===============================
def load_model():
    weights = EfficientNet_B0_Weights.IMAGENET1K_V1
    backbone = efficientnet_b0(weights=weights)

    in_features = backbone.classifier[1].in_features
    backbone.classifier = torch.nn.Sequential(
        torch.nn.Dropout(0.4),
        torch.nn.Linear(in_features, 2)
    )

    model = DeepfakeDetector.load_from_checkpoint(
        "models/best_model.ckpt",
        model=backbone,
        lr=0.0001,
        map_location="cpu"
    )

    model.eval()
    return model

model = load_model()

# ===============================
# Preprocessing
# ===============================
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# ===============================
# Extract FACE-CROPPED Frames
# ===============================
def extract_face_frames(video_path, max_frames=200):
    cap = cv2.VideoCapture(video_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS) or 25)

    faces_out = []
    frame_count = 0

    while len(faces_out) < max_frames:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count % fps == 0:  # ~1 frame per second
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.3, 5)

            if len(faces) > 0:
                x, y, w, h = faces[0]  # take first detected face
                face = frame[y:y+h, x:x+w]
                face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
                faces_out.append(Image.fromarray(face))

        frame_count += 1

    cap.release()
    return faces_out

# ===============================
# Predict VIDEO
# ===============================
def predict_video(video_path):
    faces = extract_face_frames(video_path)

    if len(faces) == 0:
        return "❌ No face detected in video", "", None

    fake_probs = []

    with torch.no_grad():
        for img in faces:
            x = preprocess(img).unsqueeze(0)
            logits = model(x)
            probs = torch.softmax(logits, dim=1)[0]
            fake_probs.append(probs[0].item())

    avg_fake = sum(fake_probs) / len(fake_probs)
    max_fake = max(fake_probs)

    final_fake = 0.6 * avg_fake + 0.4 * max_fake
    label = "🔴 Deepfake" if final_fake > 0.5 else "🟢 Real"
    confidence = max(final_fake, 1 - final_fake) * 100

    return label, f"{confidence:.2f}%", faces[0]

# ===============================
# Predict IMAGE
# ===============================
def predict_image(image_path):
    img = Image.open(image_path).convert("RGB")
    x = preprocess(img).unsqueeze(0)

    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1)[0]

    fake_prob = probs[0].item()
    real_prob = probs[1].item()

    label = "🔴 Deepfake" if fake_prob > real_prob else "🟢 Real"
    confidence = max(fake_prob, real_prob) * 100

    return label, f"{confidence:.2f}%", img

# ===============================
# Unified Predictor
# ===============================
def predict_file(file_obj):
    if file_obj is None:
        return "⚠️ No file selected", "", None

    path = file_obj.name
    mime, _ = mimetypes.guess_type(path)

    if mime and mime.startswith("image"):
        return predict_image(path)

    elif mime and mime.startswith("video"):
        return predict_video(path)

    else:
        return "❌ Unsupported file type", "", None

# ===============================
# Gradio UI
# ===============================
with gr.Blocks(title="Deepfake Detector") as demo:
    gr.Markdown(
        "## 🧠 Deepfake Detector\n"
        "Upload an **image or video**.\n\n"
        "- Images are analyzed directly\n"
        "- Videos are analyzed by sampling **multiple face frames**"
    )

    file_input = gr.File(
        label="Upload Image or Video",
        file_types=[".jpg", ".jpeg", ".png", ".mp4", ".mov"]
    )

    with gr.Row():
        prediction = gr.Textbox(label="Prediction")
        confidence = gr.Textbox(label="Confidence")

    preview = gr.Image(label="Preview (Sampled Frame / Image)")

    file_input.change(
        fn=predict_file,
        inputs=file_input,
        outputs=[prediction, confidence, preview]
    )

demo.launch()
