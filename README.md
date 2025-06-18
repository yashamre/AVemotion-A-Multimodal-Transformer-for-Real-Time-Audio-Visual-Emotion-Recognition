# AVemotion: A Multimodal Transformer for Real-Time Audio-Visual Emotion Recognition

AVemotion is an advanced deep learning project that detects human emotions by leveraging both speech and facial expression cues. Built using PyTorch, Streamlit, and SHAP for explainability, this system fuses audio and video signals via a Transformer-based model to recognize emotions with high accuracy.

---

## 📌 Table of Contents

* [Project Overview](#project-overview)
* [Dataset](#dataset)
* [Feature Extraction](#feature-extraction)
* [Model Architecture](#model-architecture)
* [Training](#training)
* [Evaluation](#evaluation)
* [Explainability](#explainability)
* [Real-Time Demo](#real-time-demo)
* [How to Run](#how-to-run)
* [Emotion Labels](#emotion-labels)
* [Applications](#applications)
* [Author](#author)
* [License](#license)

---

## 🔍 Project Overview

AVemotion classifies eight human emotions from synchronized `.wav` audio and `.mp4` video samples using a multimodal deep learning model. The combination of acoustic and visual features allows for robust emotion detection even in noisy or ambiguous settings.

---

## 📦 Dataset

**RAVDESS** - Ryerson Audio-Visual Database of Emotional Speech and Song

* 24 professional actors (12 male, 12 female)
* 8 emotions: calm, happy, sad, angry, fearful, disgust, surprised, neutral
* Audio and video clips of both speech and song

---

## 🎛 Feature Extraction

### Audio:

* Extract 40 MFCC (Mel Frequency Cepstral Coefficient) features per audio clip using `librosa`

### Video:

* Extract deep visual features using a pre-trained `ResNet18` from `torchvision`
* Sample every 30th frame from the video
* Aggregate frame features via mean pooling (512-D vector)

### Final Feature Vector:

* Concatenation of audio and video features: **552-dimensional**

---

## 🧠 Model Architecture

### `TransformerEmotionClassifier`

* **Input:** 552 features
* **Embedding Layer:** Linear + LayerNorm + ReLU + Dropout
* **Transformer Encoder:**

  * 2 layers
  * 4 attention heads
  * Feedforward size: 512
* **Classifier Head:**

  * Linear → ReLU → Dropout → Linear → 8-class output

---

## 🏋️‍♂️ Training

* Optimizer: Adam (`lr=0.001`)
* Loss: CrossEntropy with class balancing
* Epochs: 50
* Batch size: 32
* Device: CUDA/CPU
* Model checkpoint saved to `best_model.pt`

### 📉 Training Loss Plot

![Training Loss](Line%20graph.png)
This shows a steady decrease in loss, indicating the model is learning effectively across epochs.

---

## 📊 Evaluation

* Classification Report: Accuracy, Precision, Recall, F1-score
* Label-wise performance tracking
* Confusion Matrix for visual error analysis

### 📈 Accuracy per Emotion Class

![Accuracy per Class](Bar%20Plot.png)
Helps identify which emotions are learned well (e.g. 'fearful', 'disgust') and which need more data or refinement (e.g. 'sad', 'neutral').

### 🧩 Confusion Matrix

![Confusion Matrix](Confusion%20Matrix.png)
Shows where the model confuses similar emotions, especially between 'happy', 'neutral', and 'sad'.

---

## 🔎 Explainability

* SHAP (`GradientExplainer`) used to highlight feature importance
* Shows impact of both audio and video features on prediction

### 🧠 SHAP Summary Plot

![SHAP Summary](SHAP%20Summary%20Plot.png)
Ranks feature impact on model predictions, helping understand what the model relies on most.

---

## 📌 Visual Embedding Analysis

### 🌐 t-SNE Projection

![t-SNE Plot](t-SNE%20Plot.png)
Visualizes clustering of learned audio-video features. Clear separability = strong feature encoding.

---

## 🖥 Real-Time Demo

* Built with **Streamlit**
* Upload `.wav` and `.mp4` files
* Extract features in real-time
* Predict and display detected emotion
* Optional: deploy with Streamlit Cloud or Colab + LocalTunnel

---

## 🚀 How to Run

### 🧱 Install Dependencies

```bash
pip install -r requirements.txt
```

### 🧠 Train the Model (Optional)

```python
# Run notebook.ipynb or training script
```

### ▶️ Launch Streamlit App

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

## 💡 Applications

* Emotion-aware accessibility tools
* Sentiment analysis in therapy and education
* Human-robot interaction
* AI-driven customer service

---

## 🧑‍💻 Author

**Developed by:** \[Your Name]
For submission to the **Apple AIML Internship** program.

---

## 📄 License

This project is licensed for research and educational use only. For commercial or extended usage, contact the author.
