"""Run inference on a single MRI image.

Usage:
    python -m braintumor.predict --model models/vgg16_brain_tumor_best.keras --image scan.jpg
"""

from __future__ import annotations

import argparse

import numpy as np
from PIL import Image
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.models import load_model

from braintumor.data import IMG_SIZE


def preprocess_image(image: Image.Image) -> np.ndarray:
    image = image.convert("RGB").resize(IMG_SIZE)
    array = np.array(image, dtype="float32")
    array = preprocess_input(array)
    return np.expand_dims(array, axis=0)


def predict(model, image_path: str) -> tuple[str, float]:
    image = Image.open(image_path)
    batch = preprocess_image(image)
    probability = float(model.predict(batch, verbose=0)[0][0])
    label = "tumor" if probability > 0.5 else "no_tumor"
    confidence = probability if probability > 0.5 else 1 - probability
    return label, confidence


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="models/vgg16_brain_tumor_best.keras")
    parser.add_argument("--image", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model = load_model(args.model)
    label, confidence = predict(model, args.image)
    print(f"Prediction: {label} (confidence {confidence:.2%})")


if __name__ == "__main__":
    main()
