"""
This script trains a convolutional neural network (ResNet-34) to classify facial
expressions from cropped face images using transfer learning with fastai.

The dataset is expected to be organized in class-labeled folders (e.g., happy,
sad, neutral, surprise). Images are resized to 224×224 and augmented during
training. The model is fine-tuned from ImageNet-pretrained weights and exported
as a .pkl file for use in real-time inference scripts.
"""

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