"""
train_cnn_fixed.py

Stable CNN training for facial emotion classification using fastai + transfer learning.
Uses sklearn for macro-F1 to avoid fastai F1Score shape issues in multiclass.

Folder structure:
  /home/devi-amarsaikhan/MLSN-Team-6/improved_images/
      happy/
      neutral/
      sad/
      surprise/

Outputs:
  - printed accuracy + macro-F1
  - classification report + confusion matrix
  - exported model: emotion_cnn_fixed.pkl
"""

from pathlib import Path
from fastai.vision.all import *
from sklearn.metrics import classification_report, confusion_matrix, f1_score

DATA_PATH = Path("/home/devi-amarsaikhan/MLSN-Team-6/improved_images")
SEED = 42
BS = 16
IMG_SIZE = 224
EPOCHS = 8
EXPORT_NAME = "emotion_cnn_fixed.pkl"

assert DATA_PATH.exists(), f"Missing dataset folder: {DATA_PATH.resolve()}"

set_seed(SEED, reproducible=True)

# Dataloaders
item_tfms = Resize(IMG_SIZE)
batch_tfms = [
    *aug_transforms(
        do_flip=True,
        flip_vert=False,
        max_rotate=8,
        max_zoom=1.08,
        max_lighting=0.2,
        max_warp=0.02,
        p_affine=0.75,
        p_lighting=0.75,
    ),
    Normalize.from_stats(*imagenet_stats),
]

dls = ImageDataLoaders.from_folder(
    DATA_PATH,
    valid_pct=0.2,
    seed=SEED,
    item_tfms=item_tfms,
    batch_tfms=batch_tfms,
    bs=BS,
)

print("Classes:", dls.vocab, flush=True)
print("Train size:", len(dls.train_ds), "Valid size:", len(dls.valid_ds), flush=True)

# Model
learn = vision_learner(
    dls,
    resnet18,
    metrics=[accuracy],   # keep training metrics simple/stable
)

base_lr = 2e-3
print(f"Using base_lr={base_lr}", flush=True)

cbs = [
    SaveModelCallback(monitor="valid_loss", fname="best_model"),
    EarlyStoppingCallback(monitor="valid_loss", patience=3),
]

learn.fine_tune(EPOCHS, base_lr=base_lr, cbs=cbs)

# ---- Evaluation ----
preds, targs = learn.get_preds()
pred_labels = preds.argmax(dim=1)

y_true = targs.cpu().numpy()
y_pred = pred_labels.cpu().numpy()
target_names = [str(c) for c in dls.vocab]

acc = (y_pred == y_true).mean()
f1m = f1_score(y_true, y_pred, average="macro")

print(f"\nFinal Valid Accuracy: {acc:.4f}", flush=True)
print(f"Final Valid Macro-F1: {f1m:.4f}\n", flush=True)

print("Classification Report:\n", flush=True)
print(classification_report(y_true, y_pred, target_names=target_names, digits=3))

print("Confusion Matrix:\n", flush=True)
print(confusion_matrix(y_true, y_pred))

# Optional plot (only works if GUI backend available)
try:
    interp = ClassificationInterpretation.from_learner(learn)
    interp.plot_confusion_matrix(figsize=(6, 6))
except Exception as e:
    print(f"(Skipping confusion-matrix plot: {e})", flush=True)

learn.export(EXPORT_NAME)
print(f"\nSaved model to: {EXPORT_NAME}", flush=True)
