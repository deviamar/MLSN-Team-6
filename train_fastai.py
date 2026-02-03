"""

Train emotion classifier using fastai transfer learning (CNN or ViT via timm).

Expected folder structure:
  /path/to/data/
      happy/
      neutral/
      sad/
      surprise/

Outputs:
  - Prints Accuracy + Macro-F1
  - Prints sklearn classification report + confusion matrix
  - Exports a fastai Learner: .pkl
  - Saves best model weights during training: models/best.pth (fastai-managed)

Usage examples:
  # A) Stronger CNN baseline (recommended)
  python -u train_emotion_fastai.py --data /home/devi-amarsaikhan/MLSN-Team-6/improved_images \
      --arch resnet34 --epochs 10 --lr 2e-3 --bs 16 --out emotion_resnet34.pkl

  # Even stronger CNN (if you have enough compute)
  python -u train_emotion_fastai.py --data /home/devi-amarsaikhan/MLSN-Team-6/improved_images \
      --arch convnext_small --epochs 8 --lr 1e-3 --bs 8 --out emotion_convnext_small.pkl

  # B) ViT (after CNN)
  python -u train_emotion_fastai.py --data /home/devi-amarsaikhan/MLSN-Team-6/improved_images \
      --arch vit_small_patch16_224 --epochs 10 --lr 5e-4 --bs 8 --out emotion_vit_small.pkl
"""

from pathlib import Path
import argparse
import numpy as np

from fastai.vision.all import *
from sklearn.metrics import classification_report, confusion_matrix, f1_score, accuracy_score


def compute_class_weights(dls: DataLoaders):
    """
    Compute inverse-frequency class weights for CrossEntropy loss.
    Helps when some emotions (e.g., neutral) are harder / imbalanced.
    """
    # dls.train_ds items are (path,label_idx) under the hood
    # simplest: iterate train_dl once to collect targets
    y = []
    for _, t in dls.train:
        y.append(t.cpu())
    y = torch.cat(y)

    counts = torch.bincount(y, minlength=len(dls.vocab)).float()
    # Avoid divide-by-zero
    counts = torch.clamp(counts, min=1.0)
    weights = 1.0 / counts
    # Normalize weights to mean=1 (stabilizes training)
    weights = weights / weights.mean()
    return weights


def get_dls(data_dir: Path, img_size=224, bs=16, seed=42):
    set_seed(seed, reproducible=True)

    item_tfms = Resize(img_size)

    # Face-safe augmentation: modest geometry + lighting, avoid heavy warps
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
        data_dir,
        valid_pct=0.2,
        seed=seed,
        item_tfms=item_tfms,
        batch_tfms=batch_tfms,
        bs=bs,
    )
    return dls


def evaluate_sklearn(learn: Learner, dls: DataLoaders):
    preds, targs = learn.get_preds(dl=dls.valid)
    y_true = targs.cpu().numpy()
    y_pred = preds.argmax(dim=1).cpu().numpy()

    acc = accuracy_score(y_true, y_pred)
    f1m = f1_score(y_true, y_pred, average="macro")

    print("\n=== Validation Results ===", flush=True)
    print(f"Accuracy:  {acc:.4f}", flush=True)
    print(f"Macro-F1:  {f1m:.4f}", flush=True)

    target_names = [str(c) for c in dls.vocab]
    print("\nClassification Report:", flush=True)
    print(classification_report(y_true, y_pred, target_names=target_names, digits=3), flush=True)

    print("Confusion Matrix:", flush=True)
    print(confusion_matrix(y_true, y_pred), flush=True)

    # Optional fastai plot (won't work on all terminals)
    try:
        interp = ClassificationInterpretation.from_learner(learn)
        interp.plot_confusion_matrix(figsize=(6, 6))
    except Exception as e:
        print(f"(Skipping fastai confusion-matrix plot: {e})", flush=True)


def train_and_export(data_dir: str, arch: str, out: str, epochs: int, base_lr: float, bs: int, img_size: int, seed: int):
    data_dir = Path(data_dir)
    assert data_dir.exists(), f"Missing dataset folder: {data_dir.resolve()}"

    dls = get_dls(data_dir, img_size=img_size, bs=bs, seed=seed)

    print("Classes:", dls.vocab, flush=True)
    print(f"Train size: {len(dls.train_ds)} | Valid size: {len(dls.valid_ds)}", flush=True)
    print(f"Arch: {arch} | img_size={img_size} | bs={bs} | epochs={epochs} | lr={base_lr}", flush=True)

    # Class-weighted loss
    class_wts = compute_class_weights(dls).to(dls.device)
    loss_func = CrossEntropyLossFlat(weight=class_wts)

    macro_f1 = F1Score(average="macro")

    learn = vision_learner(
        dls,
        arch,
        pretrained=True,
        loss_func=loss_func,
        metrics=[accuracy, macro_f1],
    )

    cbs = [
        SaveModelCallback(monitor="valid_loss", fname="best"),
        EarlyStoppingCallback(monitor="valid_loss", patience=3),
    ]

    learn.fine_tune(epochs, base_lr=base_lr, cbs=cbs)

    evaluate_sklearn(learn, dls)

    learn.export(out)
    print(f"\nSaved model to: {out}", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="dataset folder containing class subfolders")
    ap.add_argument("--arch", default="resnet34", help="backbone arch name (resnet34, convnext_small, vit_small_patch16_224, etc.)")
    ap.add_argument("--out", default="emotion_model.pkl", help="output .pkl filename")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--img_size", type=int, default=224)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    train_and_export(args.data, args.arch, args.out, args.epochs, args.lr, args.bs, args.img_size, args.seed)
