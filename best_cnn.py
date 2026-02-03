from pathlib import Path
from fastai.vision.all import *
from sklearn.metrics import classification_report, confusion_matrix, f1_score, accuracy_score
import argparse
import numpy as np

def build_dls(data_path: Path, img_size=224, bs=16, seed=42):
    set_seed(seed, reproducible=True)

    item_tfms = Resize(img_size)
    # Face-safe augmentation: modest geometry + lighting, minimal warp
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

    return ImageDataLoaders.from_folder(
        data_path,
        valid_pct=0.2,
        seed=seed,
        item_tfms=item_tfms,
        batch_tfms=batch_tfms,
        bs=bs,
    )

def eval_sklearn(learn, dls):
    preds, targs = learn.get_preds()
    y_true = targs.cpu().numpy()
    y_pred = preds.argmax(dim=1).cpu().numpy()
    names = [str(c) for c in dls.vocab]

    acc = accuracy_score(y_true, y_pred)
    f1m = f1_score(y_true, y_pred, average="macro")

    print("\n=== Validation Results ===")
    print(f"Accuracy:  {acc:.4f}")
    print(f"Macro-F1:  {f1m:.4f}\n")

    print("Classification Report:\n")
    print(classification_report(y_true, y_pred, target_names=names, digits=3))
    print("Confusion Matrix:\n")
    print(confusion_matrix(y_true, y_pred))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="dataset folder with class subfolders")
    ap.add_argument("--arch", default="resnet34", help="resnet18/resnet34/resnet50/convnext_small/etc.")
    ap.add_argument("--out", default="emotion_best.pkl")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--img_size", type=int, default=224)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    data_path = Path(args.data)
    assert data_path.exists(), f"Missing: {data_path.resolve()}"

    dls = build_dls(data_path, img_size=args.img_size, bs=args.bs, seed=args.seed)
    print("Classes:", dls.vocab)
    print("Train size:", len(dls.train_ds), "Valid size:", len(dls.valid_ds))
    print(f"Arch: {args.arch} | img_size={args.img_size} | bs={args.bs} | epochs={args.epochs} | lr={args.lr}")

    macro_f1 = F1Score(average="macro")

    learn = vision_learner(
        dls,
        args.arch,
        pretrained=True,
        metrics=[accuracy, macro_f1],
    )

    cbs = [
        SaveModelCallback(monitor="valid_loss", fname="best"),
        EarlyStoppingCallback(monitor="valid_loss", patience=5),
    ]

    learn.fine_tune(args.epochs, base_lr=args.lr, cbs=cbs)

    eval_sklearn(learn, dls)

    learn.export(args.out)
    print(f"\nSaved model to: {args.out}")

if __name__ == "__main__":
    main()
