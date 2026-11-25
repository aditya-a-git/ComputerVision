"""Photometric perturbations in the original RGB pixel frame."""

import cv2
import numpy as np

CONDITIONS = ("daylight", "darkness", "fog", "rain", "vibration", "noise", "occlusion")


def transform(rgb, condition, strength=0.5, seed=42):
    if condition not in CONDITIONS or not 0 <= strength <= 1:
        raise ValueError("invalid condition or intensity")
    image = rgb.copy()
    if not strength or condition == "daylight":
        return image
    rng = np.random.default_rng(seed)
    h, w = image.shape[:2]
    if condition == "darkness":
        result = image * (1 - 0.95 * strength)
    elif condition == "fog":
        result = image * (1 - 0.8 * strength) + 245 * 0.8 * strength
    elif condition == "noise":
        result = image + rng.normal(0, 55 * strength, image.shape)
    elif condition == "vibration":
        length = 1 + 2 * max(1, round(9 * strength))
        kernel = np.eye(length, dtype=np.float32) / length
        return cv2.filter2D(image, -1, kernel)
    elif condition == "occlusion":
        side = np.sqrt(strength * 0.35)
        width, height = max(1, int(side * w)), max(1, int(side * h))
        x, y = rng.integers(w - width + 1), rng.integers(h - height + 1)
        image[y : y + height, x : x + width] = 35
        return image
    else:
        layer = image.copy()
        for _ in range(max(1, int(w * h * strength / 250))):
            x, y = int(rng.integers(w)), int(rng.integers(h))
            cv2.line(layer, (x, y), (x - 3, y + 10), (225, 233, 244), 1, cv2.LINE_AA)
        return cv2.addWeighted(layer, 0.65, image, 0.35, 0)
    return np.clip(result, 0, 255).astype(np.uint8)
