
# 🗣️ Speech Emotion Recognition (CNN + LSTM Hybrid Model)

🔗 **GitHub Repository**: [Speech-Emotion-Recognition](https://github.com/Masterharsh26/Speech-Emotion_Recognization)

---

## 🧠 Project Summary

| Feature            | Description                                           |
|--------------------|-------------------------------------------------------|
| **Dataset**        | RAVDESS (Ryerson Audio-Visual Database)              |
| **Emotions**       | Angry, Happy, Sad, Neutral, Fear, Disgust            |
| **Features Used**  | MFCC, Chroma, Mel Spectrogram                        |
| **Model**          | Hybrid CNN + LSTM                                    |
| **Accuracy**       | ~22% (baseline)                                      |
| **Frameworks**     | TensorFlow, Keras                                    |
| **Audio Shape**    | (40, 100, 3) pseudo-RGB spectrogram input            |

---

## 📌 Overview

This project implements a deep learning pipeline to recognize human emotions from audio speech signals using a hybrid **Convolutional Neural Network (CNN)** and **Long Short-Term Memory (LSTM)** architecture. The model classifies speech into six emotions: Angry, Happy, Sad, Neutral, Fear, and Disgust.

---

## 🧰 Workflow Summary

1. **🎧 Data Loading**  
   Load audio files from the RAVDESS dataset.

2. **🎼 Feature Extraction**  
   Extract MFCCs, Chroma, and Mel Spectrogram features to form a 3-channel spectrogram-like input.

3. **🧪 Data Augmentation**  
   Enhance model generalization with:
   - Noise injection
   - Time stretching
   - Pitch shifting

4. **🧱 Model Architecture**  
   - **CNN layers** for spatial feature extraction
   - **LSTM layers** for temporal pattern learning
   - **Dense layers** for final emotion classification

5. **📊 Training & Evaluation**  
   Evaluate with accuracy, F1-score, confusion matrix, and classification report.

---

## 🗂️ Dataset: RAVDESS

The [RAVDESS dataset](https://zenodo.org/record/1188976) contains emotional speech by professional actors. Emotions used:

- 😠 Angry  
- 😀 Happy  
- 😢 Sad  
- 😐 Neutral  
- 😨 Fear  
- 🤢 Disgust  

---

## 🎯 Model Architecture

```
Input: (40, 100, 3)
↓ Conv2D → BatchNorm → MaxPool → Dropout
↓ Conv2D → BatchNorm → MaxPool → Dropout
↓ Flatten → Reshape to sequence
↓ LSTM → Dropout → LSTM → Dropout
↓ Dense → Dropout → Dense (softmax)
Output: Emotion class (6)
```

📦 **Total Parameters**: 153,734  
📚 **Trainable**: 153,542  
📊 **Non-Trainable**: 192

---

## 🛠️ Setup Instructions

### ✅ Prerequisites

Install dependencies:

```bash
pip install tensorflow librosa scikit-learn numpy pandas matplotlib seaborn scipy
```

---

### 🚀 Getting Started

```bash
git clone https://github.com/Masterharsh26/Speech-Emotion_Recognization.git
cd Speech-Emotion_Recognization
```

1. Download and extract the [RAVDESS dataset](https://zenodo.org/record/1188976).
2. Update the dataset path in `SER.ipynb`:

```python
ravdess_path = "C:\path\to\RAVDESS"
```

3. Run all cells in the notebook to train and test the model.

---

## 📉 Results

### 🔍 Classification Report

```
               precision    recall  f1-score   support

        Angry       0.17      0.50      0.25        14
        Happy       0.14      0.12      0.13        32
          Sad       0.28      0.39      0.32        41
      Neutral       0.25      0.29      0.27        41
         Fear       0.19      0.13      0.16        38
      Disgust       1.00      0.03      0.05        37

     accuracy                           0.22       203
    macro avg       0.34      0.24      0.20       203
 weighted avg       0.36      0.22      0.20       203
```

### 📊 Confusion Matrix

![Confusion Matrix](assets/confusion_matrix.png) <!-- Replace with actual local image link -->

---

## 🎤 Predicting on New Audio

You can predict emotion from a custom `.wav` file using the model:

```python
new_audio = "path/to/your/audio.wav"
features = extract_features(new_audio)

if features is not None:
    features_reshaped = features.reshape(1, 40, 100, 3)
    prediction = model.predict(features_reshaped)
    predicted_emotion = label_encoder.inverse_transform([np.argmax(prediction)])
    print(f"Predicted Emotion: {predicted_emotion[0]}")
else:
    print("Feature extraction failed.")
```

---

## 🚧 Limitations & Future Work

- Current model performance is low due to limited training and simple architecture.
- No class balancing done (some emotion classes dominate).
- Next steps:
  - Add attention layers or transformer-based blocks
  - Try bi-directional LSTMs or GRUs
  - Use larger and multi-lingual datasets (e.g., CREMA-D, TESS)
  - Experiment with different loss functions and data augmentations

---

## 🙌 Acknowledgements

- [RAVDESS Dataset](https://zenodo.org/record/1188976)
- [LibROSA](https://librosa.org/)
- [TensorFlow](https://www.tensorflow.org/)
- [Keras](https://keras.io/)