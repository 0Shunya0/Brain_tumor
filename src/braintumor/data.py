"""Dataset loading, splitting, and augmentation for the brain tumor MRI dataset.

Expected directory layout::

    data/
        yes/   MRI images that contain a tumor
        no/    MRI images that do not contain a tumor
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.preprocessing.image import ImageDataGenerator

IMG_SIZE = (224, 224)
CLASS_NAMES = {0: "no_tumor", 1: "tumor"}


def _load_folder(folder: Path, label: int) -> tuple[np.ndarray, np.ndarray]:
    images, labels = [], []
    for path in sorted(folder.iterdir()):
        img = cv2.imread(str(path))
        if img is None:
            continue
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, IMG_SIZE)
        images.append(img)
        labels.append(label)
    return np.array(images), np.array(labels)


def load_dataset(data_dir: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """Load all images from ``data_dir/yes`` and ``data_dir/no``."""
    data_dir = Path(data_dir)
    x_tumor, y_tumor = _load_folder(data_dir / "yes", label=1)
    x_no_tumor, y_no_tumor = _load_folder(data_dir / "no", label=0)
    x = np.concatenate([x_tumor, x_no_tumor])
    y = np.concatenate([y_tumor, y_no_tumor])
    return x, y


def split_dataset(
    x: np.ndarray,
    y: np.ndarray,
    val_size: float = 0.15,
    test_size: float = 0.15,
    seed: int = 42,
):
    """Stratified train/val/test split so the class ratio is preserved in every split."""
    x_train, x_temp, y_train, y_temp = train_test_split(
        x, y, test_size=val_size + test_size, random_state=seed, stratify=y
    )
    relative_test_size = test_size / (val_size + test_size)
    x_val, x_test, y_val, y_test = train_test_split(
        x_temp, y_temp, test_size=relative_test_size, random_state=seed, stratify=y_temp
    )
    return x_train, y_train, x_val, y_val, x_test, y_test


def balance_by_oversampling(x: np.ndarray, y: np.ndarray, seed: int = 42) -> tuple[np.ndarray, np.ndarray]:
    """Duplicate minority-class samples so both classes are equally represented.

    The raw dataset is skewed ~61/39 toward "tumor" — training on it as-is is what
    previously made the model collapse to predicting "tumor" for every input. This should
    only be applied to the training split (never validation/test, which must stay
    representative of the real class distribution). Duplicated minority-class images still
    look different from each other in every epoch because the training generator applies
    random augmentation to each sample independently.

    (Keras's ``class_weight`` argument would be the usual fix, but passing it alongside a
    generator built from ``ImageDataGenerator.flow`` currently crashes in this
    TensorFlow/Keras version, so oversampling is used instead.)
    """
    rng = np.random.default_rng(seed)
    classes, counts = np.unique(y, return_counts=True)
    majority_count = counts.max()

    x_parts, y_parts = [x], [y]
    for cls, count in zip(classes, counts):
        deficit = majority_count - count
        if deficit <= 0:
            continue
        extra_idx = rng.choice(np.flatnonzero(y == cls), size=deficit, replace=True)
        x_parts.append(x[extra_idx])
        y_parts.append(y[extra_idx])

    x_balanced = np.concatenate(x_parts)
    y_balanced = np.concatenate(y_parts)
    shuffle_idx = rng.permutation(len(y_balanced))
    return x_balanced[shuffle_idx], y_balanced[shuffle_idx]


def make_generators(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    batch_size: int = 16,
):
    """Build Keras generators. Only the training split is augmented."""
    train_datagen = ImageDataGenerator(
        preprocessing_function=preprocess_input,
        rotation_range=15,
        width_shift_range=0.1,
        height_shift_range=0.1,
        shear_range=0.1,
        zoom_range=0.1,
        horizontal_flip=True,
        fill_mode="nearest",
    )
    eval_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)

    train_gen = train_datagen.flow(x_train, y_train, batch_size=batch_size)
    val_gen = eval_datagen.flow(x_val, y_val, batch_size=batch_size, shuffle=False)
    test_gen = eval_datagen.flow(x_test, y_test, batch_size=batch_size, shuffle=False)
    return train_gen, val_gen, test_gen
