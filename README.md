# Brain Tumor MRI Classifier

A binary image classifier that predicts whether a brain MRI scan shows a tumor,
using transfer learning on top of an ImageNet-pretrained VGG16.

> **Not a medical device.** This is a small educational project trained on a
> ~250-image public dataset. It is not validated for clinical use and must not
> be used to make real diagnostic decisions.

## What changed from the original version

This repo started as a notebook-only prototype with a few problems that are
fixed here:

- The model was a full VGG16 (134M params) trained **from scratch** on 253
  images with no class balancing. It collapsed to predicting "tumor" for
  every input (61% accuracy — exactly the dataset's tumor/no-tumor ratio, i.e.
  no better than guessing the majority class).
- `gui.py` didn't even load the brain tumor model — it trained a separate
  model on CIFAR-10 and classified images into 10 unrelated categories.
- Absolute, machine-specific paths were hardcoded into the notebook.
- Three duplicate 1.6GB `.keras` files and a duplicated copy of the dataset
  were sitting in the repo root.

This version instead:

- Uses a **frozen, pretrained VGG16 base** with a small trainable
  classification head — far fewer parameters to overfit with, and it starts
  from real, generalizable image features instead of random weights.
- Applies **minority-class oversampling** and **data augmentation** to correct
  for the ~61/39 tumor/no-tumor skew in the dataset (Keras's `class_weight`
  would be the more usual fix, but it currently crashes when combined with an
  `ImageDataGenerator` generator in this TensorFlow/Keras version).
- Optionally fine-tunes the last few convolutional layers after the head has
  converged.
- Splits code out of the notebook into a proper `src/braintumor` package with
  a training CLI, a prediction CLI, and a GUI that actually uses the trained
  model.

## Project structure

```
src/braintumor/
    data.py      # dataset loading, stratified split, augmentation, class weights
    model.py      # VGG16 transfer-learning model definition
    train.py      # training CLI
    predict.py    # single-image inference CLI
    gui.py        # Tkinter app for interactive predictions
tests/
    test_model.py # smoke test for model construction
notebooks/
    BrainTumorDetection.ipynb  # original exploratory notebook, kept for reference
data/            # not tracked in git — put yes/ and no/ image folders here
models/          # not tracked in git — trained .keras files land here
```

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate        # on Windows
pip install -r requirements.txt
pip install -e .
```

## Dataset

Download the [Brain MRI Images for Brain Tumor Detection](https://www.kaggle.com/datasets/navoneel/brain-mri-images-for-brain-tumor-detection)
dataset (or your own labeled MRI images) and arrange it as:

```
data/
    yes/   # MRI images that contain a tumor
    no/    # MRI images that do not
```

## Training

```bash
python -m braintumor.train --data-dir data --output-dir models --epochs 30 --fine-tune
```

This trains the classifier head with the base frozen, optionally fine-tunes
the last few VGG16 layers, then prints a confusion matrix and classification
report on a held-out test split. The best checkpoint (by validation accuracy)
is saved to `models/vgg16_brain_tumor_best.keras`.

## Predicting on a single image

```bash
python -m braintumor.predict --model models/vgg16_brain_tumor_best.keras --image path/to/scan.jpg
```

## GUI

```bash
python -m braintumor.gui --model models/vgg16_brain_tumor_best.keras
```

## Tests

```bash
python -m unittest discover tests
```

## Known limitations

- The dataset is tiny (253 images total), so reported metrics have wide
  variance between runs and don't reflect clinical-grade performance.
- ImageNet features are a reasonable but imperfect prior for MRI images —
  domain-specific pretraining (e.g. on radiology datasets) would likely help.
- No cross-validation is performed; a single stratified train/val/test split
  is used.
