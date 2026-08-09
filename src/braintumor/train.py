"""Train the brain tumor classifier.

Usage:
    python -m braintumor.train --data-dir data --output-dir models --epochs 30 --fine-tune
"""

from __future__ import annotations

import argparse
from pathlib import Path

from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

from braintumor.data import balance_by_oversampling, load_dataset, make_generators, split_dataset
from braintumor.model import build_model, unfreeze_for_fine_tuning


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default="data", help="Directory containing yes/ and no/ subfolders")
    parser.add_argument("--output-dir", default="models", help="Where to save trained model files")
    parser.add_argument("--epochs", type=int, default=30, help="Epochs for the frozen-base phase")
    parser.add_argument("--fine-tune-epochs", type=int, default=15, help="Epochs for the fine-tuning phase")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--fine-tune",
        action="store_true",
        help="After training the head, unfreeze the last few VGG16 layers and continue at a lower LR",
    )
    return parser.parse_args()


def evaluate(model, test_gen, y_test) -> None:
    probs = model.predict(test_gen)
    preds = (probs > 0.5).astype("int32").ravel()
    print("\nConfusion matrix (rows=true, cols=pred), classes = [no_tumor, tumor]:")
    print(confusion_matrix(y_test, preds))
    print("\nClassification report:")
    print(classification_report(y_test, preds, target_names=["no_tumor", "tumor"], zero_division=0))


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading images from {args.data_dir} ...")
    x, y = load_dataset(args.data_dir)
    print(f"Loaded {len(x)} images ({int(y.sum())} tumor, {len(y) - int(y.sum())} no-tumor).")

    x_train, y_train, x_val, y_val, x_test, y_test = split_dataset(x, y, seed=args.seed)
    print(f"Train/val/test sizes: {len(x_train)}/{len(x_val)}/{len(x_test)}")

    x_train, y_train = balance_by_oversampling(x_train, y_train, seed=args.seed)
    print(f"Balanced training set to {len(y_train)} samples ({int(y_train.sum())} tumor, "
          f"{len(y_train) - int(y_train.sum())} no-tumor) by oversampling the minority class.")

    train_gen, val_gen, test_gen = make_generators(
        x_train, y_train, x_val, y_val, x_test, y_test, batch_size=args.batch_size
    )

    model, base = build_model()
    checkpoint_path = output_dir / "vgg16_brain_tumor_best.keras"
    callbacks = [
        ModelCheckpoint(str(checkpoint_path), save_best_only=True, monitor="val_accuracy", mode="max"),
        EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True),
    ]

    print("\n--- Phase 1: training classifier head (base frozen) ---")
    model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=args.epochs,
        callbacks=callbacks,
    )

    if args.fine_tune:
        print("\n--- Phase 2: fine-tuning last VGG16 layers ---")
        unfreeze_for_fine_tuning(base, num_layers=4)
        model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
        model.optimizer.learning_rate.assign(1e-5)
        model.fit(
            train_gen,
            validation_data=val_gen,
            epochs=args.fine_tune_epochs,
            callbacks=callbacks,
        )

    print("\n--- Evaluating on held-out test set ---")
    evaluate(model, test_gen, y_test)

    final_path = output_dir / "vgg16_brain_tumor_final.keras"
    model.save(final_path)
    print(f"\nSaved best checkpoint to {checkpoint_path}")
    print(f"Saved final model to {final_path}")


if __name__ == "__main__":
    main()
