import sys
import os
sys.path.append(os.path.abspath(".."))

from models.colorization.co_model_1 import ColorizeModel
import cv2

class Colorize:
    def __init__(self, image, model_type="color_model1", saturation_factor=1.25, force_recolor=False):
        self.img = image
        self.saturation_factor = saturation_factor
        self.force_recolor = force_recolor
        if model_type == "color_model1":
            self.model = ColorizeModel()  # Load model from models/
        else:
            self.model = ColorizeModel()  # Placeholder for other models

    def _boost_saturation(self, image):
        """Boost saturation in HSV space while keeping values in valid range."""
        if image is None or len(image.shape) != 3:
            return image

        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV).astype("float32")
        hsv[:, :, 1] = hsv[:, :, 1] * float(self.saturation_factor)
        hsv[:, :, 1] = hsv[:, :, 1].clip(0, 255)
        hsv = hsv.astype("uint8")
        return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    def apply_colorize(self):
        """Apply colorization using model from ../models"""
        result = self.model.colorize(self.img, force_recolor=self.force_recolor)
        result = self._boost_saturation(result)
        return result
