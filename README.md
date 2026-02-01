# Facial Emotion Detection (MLSN Team 6)

This project implements facial emotion recognition using two approaches:

1. **Baseline model:** Random Forest classifier trained on flattened face images
2. **Improved model:** Convolutional Neural Network (CNN) using transfer learning (ResNet via FastAI)

The system supports:

* Offline training and evaluation
* Live webcam emotion prediction
* Model export and reuse

---

## Project Structure

```
MLSN-Team-6/
│
├── images/                 # Original dataset
├── faces_cropped/          # Face-cropped dataset (generated)
├── improved_images/        # Alternative dataset
│
├── facialEmotionClassifier.ipynb   # Random Forest baseline
├── CnnEmotionClassifier.ipynb      # CNN (FastAI)
├── prep_faces.py                  # Face cropping preprocessing script
├── runModel.py                    # Webcam (Random Forest)
├── runModelCnn.py                 # Webcam (CNN)
│
├── emotion_detection_model.pkl    # Saved Random Forest model
├── emotion_cnn.pkl                # Saved CNN model
└── README.md
```

---

## Environment Setup (All OS)

### 1. Clone the repository

```bash
git clone <your-repo-url>
cd MLSN-Team-6
```

---

## Virtual Environment Setup

### Windows (PowerShell)

```powershell
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

### Linux / macOS

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

If `requirements.txt` does not exist, install manually:

```bash
pip install fastai torch torchvision opencv-python cvlib scikit-learn seaborn matplotlib pandas numpy
```

---

## Face Preprocessing (Important)

To improve model performance, training images are cropped using the same face detector used during webcam inference.

Run:

```bash
python prep_faces.py
```

This will create:

```
faces_cropped/
  happy/
  neutral/
  sad/
  surprise/
```

This step ensures training data matches live webcam input.

---

## Train Baseline Model (Random Forest)

Open:

```
facialEmotionClassifier.ipynb
```

Run all cells.
This will:

* Load images
* Extract features
* Train RandomForest
* Save model to:

```
emotion_detection_model.pkl
```

---

## Train CNN Model (FastAI)

Open:

```
CnnEmotionClassifier.ipynb
```

Ensure dataset path points to:

```python
path = Path("faces_cropped")
```

Train model:

```python
learn = vision_learner(dls, resnet34, metrics=[accuracy])
learn.fine_tune(8)
learn.export("emotion_cnn.pkl")
```

---

## Evaluate Model Performance

After training:

```python
interp = ClassificationInterpretation.from_learner(learn)
interp.plot_confusion_matrix()

preds, targs = learn.get_preds()
print("Accuracy:", accuracy(preds, targs).item())
```

Metrics to report:

* Accuracy
* Macro F1-score
* Confusion matrix

---

## Run Webcam Emotion Detection

### Random Forest:

```bash
python runModel.py
```

### CNN:

```bash
python runModelCnn.py
```

Press `q` to exit.

---

## Technologies Used

* Python 3.10+
* OpenCV / cvlib
* Scikit-learn
* FastAI / PyTorch
* Seaborn / Matplotlib
