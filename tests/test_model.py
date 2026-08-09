"""Smoke test: the model builds and produces the expected output shape.

Run with: python -m unittest discover tests
"""

import unittest

import numpy as np

from braintumor.model import build_model


class BuildModelTest(unittest.TestCase):
    def test_output_shape_and_frozen_base(self):
        model, base = build_model()
        self.assertFalse(base.trainable)

        dummy_batch = np.zeros((2, 224, 224, 3), dtype="float32")
        predictions = model.predict(dummy_batch, verbose=0)

        self.assertEqual(predictions.shape, (2, 1))
        self.assertTrue(((predictions >= 0) & (predictions <= 1)).all())


if __name__ == "__main__":
    unittest.main()
