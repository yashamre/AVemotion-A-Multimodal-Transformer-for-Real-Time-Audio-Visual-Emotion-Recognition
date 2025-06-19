import streamlit as st
import torch
import torchaudio
import torchvision.models as models
import torchvision.transforms as transforms
import numpy as np
import cv2
import librosa
import tempfile
from pathlib import Path

# ----- Transformer Model Definition -----
class TransformerEmotionClassifier(torch.nn.Module):
    def __init__(self, input_dim=552, hidden_dim=256, num_heads=4, num_classes=8):
        super().__init__()
        self.embedding = torch.nn.Sequential(
            torch.nn.Linear(input_dim, hidden_dim),
            torch.nn.LayerNorm(hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.2)
        )
        encoder_layer = torch.nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=num_heads, dim_feedforward=512)
        self.transformer = torch.nn.TransformerEncoder(encoder_layer, num_layers=2)
        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(hidden_dim, 128),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.2),
            torch.nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.embedding(x.unsqueeze(1))
        x = self.transformer(x).mean(dim=1)
        return self.classifier(x)

# ----- Helper Functions -----
@st.cache_resource
def load_model():
    model = TransformerEmotionClassifier()
    model.load_state_dict(torch.load("best_model.pt", map_location=torch.device('cpu')))
    model.eval()
    return model

def extract_audio_features(audio_file):
    y, sr = librosa.load(audio_file, sr=22050)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
    mfcc_mean = np.mean(mfcc.T, axis=0)
    return mfcc_mean

def extract_video_features(video_file):
    model = models.resnet18(pretrained=True)
    model = torch.nn.Sequential(*list(model.children())[:-1])
    model.eval()
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])

    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_file:
        tmp_file.write(video_file.read())
        tmp_path = tmp_file.name

    cap = cv2.VideoCapture(tmp_path)
    frames = []
    i = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if i % 30 == 0:
            img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img_tensor = transform(img).unsqueeze(0)
            with torch.no_grad():
                features = model(img_tensor).squeeze().numpy()
            frames.append(features)
        i += 1
    cap.release()
    return np.mean(frames, axis=0) if frames else np.zeros(512)

# ----- Streamlit UI -----
st.set_page_config(page_title='AVemotion', layout='centered')
st.title("AVemotion: Real-Time Emotion Recognition")
st.write("Upload your audio and video files for emotion detection.")

audio_file = st.file_uploader("Upload Audio (.wav)", type=['wav'])
video_file = st.file_uploader("Upload Video (.mp4)", type=['mp4'])

if audio_file and video_file:
    with st.spinner("Extracting features and predicting..."):
        audio_feat = extract_audio_features(audio_file)
        video_feat = extract_video_features(video_file)
        combined_feat = np.concatenate((audio_feat, video_feat))
        input_tensor = torch.tensor(combined_feat, dtype=torch.float32).unsqueeze(0)
        model = load_model()
        with torch.no_grad():
            pred = model(input_tensor)
            pred_label = torch.argmax(pred, dim=1).item()

        emotions = ['angry', 'calm', 'disgust', 'fearful', 'happy', 'neutral', 'sad', 'surprised']
        st.success(f"Predicted Emotion: **{emotions[pred_label]}**")