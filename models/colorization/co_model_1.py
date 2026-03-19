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

        resolved_model_path = self._resolve_model_path(model_path)
        self.has_trained_weights = resolved_model_path is not None
        if self.has_trained_weights:
            self.net.load_state_dict(torch.load(resolved_model_path, map_location=self.device))

        self.input_size = (256, 256)

    def _resolve_model_path(self, model_path):
        """Resolve a valid checkpoint path, if available."""
        if model_path and os.path.exists(model_path):
            return model_path

        base_dir = os.path.dirname(os.path.abspath(__file__))
        candidates = [
            os.path.join(base_dir, "color_model1.pth"),
            os.path.join(base_dir, "co_model_1.pth"),
            os.path.join(base_dir, "checkpoint.pth"),
            os.path.join(base_dir, "checkpoints", "color_model1.pth"),
            os.path.join(base_dir, "checkpoints", "co_model_1.pth"),
        ]

        for path in candidates:
            if os.path.exists(path):
                return path

        return None

    def _to_gray(self, img):
        if len(img.shape) == 3:
            return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return img

    def _is_already_colored(self, img):
        """Detect natural color images so we don't recolor them into a global tint."""
        if img is None or len(img.shape) != 3:
            return False

        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        mean_saturation = float(np.mean(hsv[:, :, 1]))

        b, g, r = cv2.split(img)
        channel_difference = float(
            np.mean(np.abs(r.astype(np.float32) - g.astype(np.float32)))
            + np.mean(np.abs(g.astype(np.float32) - b.astype(np.float32)))
            + np.mean(np.abs(r.astype(np.float32) - b.astype(np.float32)))
        )

        return mean_saturation > 12.0 and channel_difference > 10.0

    def _is_chroma_collapsed(self, ab_output):
        """Detect degenerate predictions where ab channels are nearly constant."""
        ab = ab_output.squeeze(0).permute(1, 2, 0).detach().cpu().numpy()
        std_a = float(np.std(ab[:, :, 0]))
        std_b = float(np.std(ab[:, :, 1]))
        return std_a < 0.03 and std_b < 0.03

    def _grayscale_fallback(self, img):
        """Safe fallback: keep neutral tones instead of introducing wrong global colors."""
        gray = self._to_gray(img)
        gray = cv2.equalizeHist(gray)
        return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

    def preprocess(self, img):
        """Convert input image to normalized grayscale tensor."""
        gray = self._to_gray(img)

        gray_resized = cv2.resize(gray, self.input_size)
        gray_normalized = (gray_resized.astype(np.float32) / 127.5) - 1.0
        
        # Convert to tensor with batch dimension
        tensor = torch.from_numpy(gray_normalized).unsqueeze(0).unsqueeze(0)
        return tensor.to(self.device)
    
    def colorize(self, img, force_recolor=False):
        """Colorize a grayscale image"""
        if (not force_recolor) and self._is_already_colored(img):
            return img.copy()

        if not self.has_trained_weights:
            return self._grayscale_fallback(img)

        with torch.no_grad():
            # Preprocess
            tensor = self.preprocess(img)
            
            # Run model
            ab = self.net(tensor)  # Output: [1, 2, H, W]

            if self._is_chroma_collapsed(ab):
                return self._grayscale_fallback(img)
            
            # Postprocess
            result = self.postprocess(img, ab)
        
        return result
    
    def postprocess(self, original_img, ab_output):
        """Merge L and predicted ab channels, then convert Lab to BGR."""
        gray = self._to_gray(original_img)

        gray_resized = cv2.resize(gray, self.input_size)

        ab = ab_output.squeeze(0).permute(1, 2, 0).cpu().numpy()
        ab = ((ab + 1) * 127.5).astype(np.uint8)

        l_channel = gray_resized[:, :, np.newaxis]
        lab = np.concatenate([l_channel, ab], axis=2)

        colorized = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        colorized = cv2.resize(colorized, (original_img.shape[1], original_img.shape[0]))
        return colorized
