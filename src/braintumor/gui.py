"""Tkinter GUI for the brain tumor classifier.

The original gui.py in this repo actually trained and ran a CIFAR-10 model
unrelated to brain tumors. This version loads the trained tumor-detection
model and classifies MRI images selected by the user.

Usage:
    python -m braintumor.gui --model models/vgg16_brain_tumor_best.keras
"""

from __future__ import annotations

import argparse
import tkinter as tk
from tkinter import filedialog, messagebox

from PIL import Image, ImageTk
from tensorflow.keras.models import load_model

from braintumor.predict import predict


class BrainTumorApp:
    def __init__(self, model):
        self.model = model

        self.window = tk.Tk()
        self.window.title("Brain Tumor Detection")
        self.window.geometry("400x500")

        self.label = tk.Label(self.window, text="Select an MRI image for prediction:")
        self.label.pack(pady=10)

        self.button = tk.Button(self.window, text="Open Image", command=self.open_image, bg="lightblue")
        self.button.pack(pady=20)

        self.image_label = tk.Label(self.window)
        self.image_label.pack(pady=10)

        self.result_label = tk.Label(self.window, text="", font=("Helvetica", 14))
        self.result_label.pack(pady=10)

    def open_image(self) -> None:
        file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg *.jpeg *.png")])
        if not file_path:
            return
        self.display_image(file_path)
        self.predict_image(file_path)

    def display_image(self, image_path: str) -> None:
        img = Image.open(image_path).resize((224, 224), Image.Resampling.LANCZOS)
        photo = ImageTk.PhotoImage(img)
        self.image_label.configure(image=photo)
        self.image_label.image = photo

    def predict_image(self, image_path: str) -> None:
        try:
            label, confidence = predict(self.model, image_path)
        except Exception as exc:  # noqa: BLE001 - surface any load/predict error to the user
            messagebox.showerror("Prediction failed", str(exc))
            return
        text = "Tumor detected" if label == "tumor" else "No tumor detected"
        self.result_label.config(text=f"{text} ({confidence:.1%} confidence)")

    def run(self) -> None:
        self.window.mainloop()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="models/vgg16_brain_tumor_best.keras")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model = load_model(args.model)
    BrainTumorApp(model).run()


if __name__ == "__main__":
    main()
