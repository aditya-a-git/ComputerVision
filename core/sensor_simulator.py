import cv2


def proxy(rgb):
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    enhanced = cv2.createCLAHE(2.0, (8, 8)).apply(gray)
    return cv2.cvtColor(enhanced, cv2.COLOR_GRAY2RGB)
