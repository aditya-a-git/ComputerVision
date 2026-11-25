import numpy as np
import cv2


def quality(rgb):
    """Illustrative brightness, contrast and sharpness score in [0,1]."""
    gray = cv2.cvtColor(rgb,cv2.COLOR_RGB2GRAY)
    brightness = 1-abs(gray.mean()-127.5)/127.5
    contrast = min(1,gray.std()/48)
    sharpness = min(1,cv2.Laplacian(gray,cv2.CV_32F).var()/120)
    return float(np.clip((brightness+contrast+sharpness)/3,0,1))
