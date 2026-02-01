from fastprogress import force_console_behavior
force_console_behavior()

from fastai.vision.all import *

PATH = Path("faces_cropped")   # or faces_cropped/ if that’s your folder
dls = ImageDataLoaders.from_folder(
    PATH,
    valid_pct=0.2,
    seed=42,
    item_tfms=Resize(224),
    batch_tfms=aug_transforms(mult=1.0)
)

learn = vision_learner(dls, resnet34, metrics=[accuracy])
learn.fine_tune(8)

learn.export("emotion_cnn.pkl")
print("Saved: emotion_cnn.pkl")
