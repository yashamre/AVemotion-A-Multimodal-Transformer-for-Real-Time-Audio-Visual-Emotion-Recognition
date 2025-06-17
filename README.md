# AVemotion: A Multimodal Transformer for Real-Time Audio-Visual Emotion Recognition

AVemotion is a deep learning-based project that combines audio and visual cues to detect human emotions in real-time. Built using PyTorch and Streamlit, this model leverages MFCC features from speech and deep features from facial video using ResNet18, fused through a Transformer-based attention model. Ideal for applications in accessibility support, human-computer interaction, and affective computing.

---

## 🔍 Project Overview

* **Goal:** Accurately classify emotions from synchronized audio and video inputs
* **Dataset:** RAVDESS (Ryerson Audio-Visual Database of Emotional Speech and Song)
* **Modalities:**

  * Audio (.wav) → MFCC features
  * Video (.mp4) → ResNet18 features

---

## 📁 Project Structure

```
.
├── app.py               # Streamlit interface
├── best_model.pt        # Trained Transformer model
├── data/                # RAVDESS audio-video files (unzipped)
├── notebook.ipynb       # Main development notebook
└── README.md            # Project overview
```

---

## 🧠 Model Architecture

**TransformerEmotionClassifier:**

* Input: 552 features (40 audio + 512 video)
* Embedding layer + LayerNorm + Dropout
* Transformer Encoder (2 layers, 4 heads)
* Fully connected classifier

---

## 🚀 How to Run

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Train the Model (Optional)

```python
# inside notebook.ipynb
# runs 50 epochs and saves best_model.pt
```

### 3. Launch the Streamlit App

```bash
streamlit run app.py
```

---

## 🎭 Emotion Labels

```
0: angry
1: calm
2: disgust
3: fearful
4: happy
5: neutral
6: sad
7: surprised
```

---

## 🔬 Explainability

Model insights are generated using SHAP GradientExplainer, visualizing feature importance from both audio and video domains.

---

## 💡 Applications

* Accessibility tools (emotion feedback for hearing/speech impaired)
* Virtual assistants and robotics
* Interactive learning and therapy platforms

## 📄 License

This project is for educational and research use. For other use cases, please contact the author.
