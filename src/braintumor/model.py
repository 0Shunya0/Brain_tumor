"""Model definition: a VGG16 transfer-learning head for binary tumor classification.

The original version of this project trained a full VGG16 (~134M params) from
scratch on 253 images, which just memorized the majority class. With that little
data, a frozen ImageNet-pretrained convolutional base plus a small trainable head
is far more likely to generalize.
"""

from __future__ import annotations

from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.applications import VGG16


def build_model(input_shape: tuple[int, int, int] = (224, 224, 3), learning_rate: float = 1e-4):
    base = VGG16(weights="imagenet", include_top=False, input_shape=input_shape)
    base.trainable = False

    model = models.Sequential(
        [
            base,
            layers.GlobalAveragePooling2D(),
            layers.Dense(256, activation="relu"),
            layers.Dropout(0.5),
            layers.Dense(1, activation="sigmoid"),
        ]
    )
    model.compile(
        optimizer=optimizers.Adam(learning_rate=learning_rate),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )
    return model, base


def unfreeze_for_fine_tuning(base, num_layers: int = 4) -> None:
    """Unfreeze the last ``num_layers`` layers of the base model for fine-tuning."""
    base.trainable = True
    for layer in base.layers[:-num_layers]:
        layer.trainable = False
