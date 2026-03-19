import sys
import os
sys.path.append(os.path.abspath(".."))

from models.colorization.co_model_1 import ColorizeModel

class Colorize:
    def __init__(self, image, model_type="color_model1"):
        self.img = image
        if model_type == "color_model1":
            self.model = ColorizeModel()  # Load model from models/
        else:
            self.model = ColorizeModel()  # Placeholder for other models

    def apply_colorize(self):
        """Apply colorization using model from ../models"""
        result = self.model.colorize(self.img)
        return result
