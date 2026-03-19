import os

import cv2
import numpy as np
import torch
import torch.nn as nn


class ColorizeNet(nn.Module):
    """PyTorch neural network for predicting Lab ab channels."""

    def __init__(self):
        super().__init__()

        self.enc1 = nn.Conv2d(1, 64, kernel_size=3, padding=1)
        self.enc2 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)
        self.enc3 = nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1)

        self.dec1 = nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1)
        self.dec2 = nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1)
        self.dec3 = nn.Conv2d(64, 2, kernel_size=3, padding=1)

        self.relu = nn.LeakyReLU()
        self.tanh = nn.Tanh()

    def forward(self, x):
        x = self.relu(self.enc1(x))
        x = self.relu(self.enc2(x))
        x = self.relu(self.enc3(x))

        x = self.relu(self.dec1(x))
        x = self.relu(self.dec2(x))
        x = self.tanh(self.dec3(x))
        return x


class ColorizeModel:
    """Loads the model and runs colorization inference."""

    def __init__(self, model_path=None, device="cpu"):
        self.device = torch.device(device)
        self.net = ColorizeNet().to(self.device)
        self.net.eval()

        if model_path and os.path.exists(model_path):
            self.net.load_state_dict(torch.load(model_path, map_location=self.device))

        self.input_size = (256, 256)

    def preprocess(self, img):
        """Convert input image to normalized grayscale tensor."""
        if len(img.shape) == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img

        gray_resized = cv2.resize(gray, self.input_size)
        gray_normalized = (gray_resized.astype(np.float32) / 127.5) - 1.0
        
        # Convert to tensor with batch dimension
        tensor = torch.from_numpy(gray_normalized).unsqueeze(0).unsqueeze(0)
        return tensor.to(self.device)
    
    def colorize(self, img):
        """Colorize a grayscale image"""
        with torch.no_grad():
            # Preprocess
            tensor = self.preprocess(img)
            
            # Run model
            ab = self.net(tensor)  # Output: [1, 2, H, W]
            
            # Postprocess
            result = self.postprocess(img, ab)
        
        return result
    
    def postprocess(self, original_img, ab_output):
        """Merge L and predicted ab channels, then convert Lab to BGR."""
        if len(original_img.shape) == 3:
            gray = cv2.cvtColor(original_img, cv2.COLOR_BGR2GRAY)
        else:
            gray = original_img

        gray_resized = cv2.resize(gray, self.input_size)

        ab = ab_output.squeeze(0).permute(1, 2, 0).cpu().numpy()
        ab = ((ab + 1) * 127.5).astype(np.uint8)

        l_channel = gray_resized[:, :, np.newaxis]
        lab = np.concatenate([l_channel, ab], axis=2)

        colorized = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        colorized = cv2.resize(colorized, (original_img.shape[1], original_img.shape[0]))
        return colorized
