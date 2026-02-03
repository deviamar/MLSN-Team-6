import timm

learn_vit = vision_learner(
    dls,
    "vit_small_patch16_224",
    pretrained=True,
    metrics=[accuracy, F1Score(average="macro")]
)

# Transformers often like slightly smaller LR
learn_vit.fine_tune(8, base_lr=1e-3)
eval_and_export(learn_vit, "emotion_vit_small_patch16_224.pkl")
