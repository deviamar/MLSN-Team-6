# Facial Emotion Classification (MLSN Team 6)

This project builds and evaluates facial emotion classifiers using progressively stronger models under limited data and compute. Starting from a noisy Kaggle dataset, we manually curated and relabeled images into four emotions (**happy, neutral, sad, surprise**), then compared:

1. **Baseline:** Random Forest (traditional ML)
2. **Deep Learning:** CNN with transfer learning (ResNet, FastAI)
3. **Transformer:** Vision Transformer (ViT)

The best model (ViT) achieved approximately **0.78 accuracy** and **0.77 macro-F1**. A tuned CNN (ResNet-34) achieved **0.764 accuracy** and **0.755 macro-F1**, while a Random Forest baseline reached **0.540 accuracy** and **0.530 macro-F1**. Most remaining errors occur between **neutral** and **sad** expressions.

The system supports:

* Offline training and evaluation
* Confusion-matrix and F1 reporting
* Webcam-based emotion inference
* Model export and reuse

---

## Project Structure

```
MLSN-Team-6/
│
├── images/                     # Raw Kaggle dataset
├── faces_cropped/              # Face-cropped dataset (generated)
├── improved_images/            # Manually curated final dataset
│
├── facialEmotionClassifier.ipynb   # Random Forest baseline
├── CnnEmotionClassifier.ipynb      # CNN (FastAI)
│
├── train_cnn_fixed.py          # Stable CNN training script
├── train_fastai.py             # General CNN training framework
├── vit.py                      # Vision Transformer training
├── best_cnn.py                 # Best-performing CNN config
│
├── prep_faces.py               # Face cropping preprocessing
├── runModel.py                 # Webcam (Random Forest)
├── runModelCnn.py              # Webcam (CNN)
├── runModelSEP.py              # Alternate webcam pipeline
│
├── emotion_detection_model.pkl # Saved Random Forest model
├── emotion_cnn_fixed.pkl       # Saved CNN model
├── emotion_vit.pkl             # Saved ViT model
│
├── program.py                  # Shared utilities
├── README.md
└── LICENSE
```

---

## Environment Setup

### Clone the repository

```bash
git clone <your-repo-url>
cd MLSN-Team-6
```

---

## Virtual Environment

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

If `requirements.txt` is missing:

```bash
pip install fastai torch torchvision timm opencv-python cvlib scikit-learn seaborn matplotlib pandas numpy
```

---

## Dataset Preparation

### Manual Curation

We manually reviewed and filtered the Kaggle dataset, retaining approximately **700 images per class** and removing:

* watermarks or large text
* incorrect or ambiguous emotion labels
* non-faces, heavy occlusion, or extreme blur

Final structure:

```
improved_images/
  happy/
  neutral/
  sad/
  surprise/
```

---

## Face Cropping (Optional but Recommended)

To match webcam input, crop faces before training:

```bash
python prep_faces.py
```

This generates:

```
faces_cropped/
  happy/
  neutral/
  sad/
  surprise/
```

---

## Train Models

### 1) Random Forest (Baseline)

Open:

```
facialEmotionClassifier.ipynb
```

Run all cells. This will:

* Load images
* Extract features
* Train Random Forest
* Save model to:

```
emotion_detection_model.pkl
```

Baseline result:

* Accuracy ≈ **0.540**
* Macro-F1 ≈ **0.530**

This baseline highlights the limitations of pixel-level features for subtle facial expression discrimination.

---

### 2) CNN (ResNet, FastAI)

Run:

```bash
python train_cnn_fixed.py
```

or train interactively with:

```
CnnEmotionClassifier.ipynb
```

Exports:

```
emotion_cnn_fixed.pkl
```

Results:

* Naive fine-tuning peaked at **0.631 accuracy** before degrading due to overfitting.
* Tuned model (curated data + face-safe augmentation + early stopping):

  * Accuracy ≈ **0.764**
  * Macro-F1 ≈ **0.755**

---

### 3) Vision Transformer (ViT)

Run:

```bash
python vit.py
```

Exports:

```
emotion_vit.pkl
```

Best ViT result:

* Accuracy ≈ **0.780**
* Macro-F1 ≈ **0.771**

---

## Performance Summary

| Model                            | Accuracy         | Macro-F1  | Notes                                                          |
| -------------------------------- | ---------------- | --------- | -------------------------------------------------------------- |
| Random Forest (baseline)         | **0.540**        | **0.530** | Flattened pixel features; struggles with subtle expressions    |
| CNN (ResNet-34, naive fine-tune) | **0.631 (peak)** | ~0.55     | Started near chance (0.20) and overfit with continued training |
| CNN (ResNet-34, tuned)           | **0.764**        | **0.755** | Curated data + face-safe augmentation + early stopping         |
| Vision Transformer (ViT)         | **0.780**        | **0.771** | Best overall performance                                       |

Notable class-level result (tuned CNN / ViT):

* **Happy:** Precision ≈ **0.82**, Recall ≈ **0.88** (F1 ≈ **0.85**)

---

## Evaluation

Metrics reported:

* Accuracy
* Macro-F1
* Confusion Matrix

Example:

```python
interp = ClassificationInterpretation.from_learner(learn)
interp.plot_confusion_matrix()

preds, targs = learn.get_preds()
print("Accuracy:", accuracy(preds, targs).item())
```

Consistent error pattern:

* **neutral ↔ sad confusion**

---

## Run Webcam Emotion Detection

### Random Forest

```bash
python runModel.py
```

### CNN

```bash
python runModelCnn.py
```

### Alternate pipeline

```bash
python runModelSEP.py
```

Press `q` to exit.

---

## Technologies Used

* Python 3.10+
* OpenCV / cvlib
* Scikit-learn
* FastAI / PyTorch
* timm (ViT backbones)
* NumPy / Pandas
* Matplotlib / Seaborn

---

## Key Takeaways

- The largest performance gains came from **dataset curation and preprocessing**, rather than from changing architectures.  
- A tuned ResNet substantially outperformed a naive fine-tuning run, showing that **training protocol mattered more than model size** in this regime.  
- Vision Transformers achieved the strongest overall results, but the improvement over a well-trained CNN was incremental, suggesting that expression ambiguity is now the dominant bottleneck.  
- Persistent confusion between neutral and sad indicates that future progress likely depends more on **better labeling criteria or richer data** than on deeper networks alone.
