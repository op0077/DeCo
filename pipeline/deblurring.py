import cv2
from PIL import Image

class Deblurr:
    def __init__(self,image):
        self.img = image

    def apply_deblur(self):
        self.img = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)
        return self.img