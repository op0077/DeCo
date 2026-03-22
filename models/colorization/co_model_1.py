import os
import argparse
from pathlib import Path

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
    """Loads the model and runs colorization inference.
    If no trained weights are found, it will gracefully degrade to a grayscale fallback.
    L (lightness/grayscale) in → model → ab (color) out
    """

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


def train_colorize_net(
    epochs=10,
    steps_per_epoch=500,
    batch_size=16,
    learning_rate=1e-4,
    image_size=256,
    num_workers=0,
    device=None,
    checkpoint_dir=None,
    checkpoint_name="co_model_1.pth",
    log_interval=10,
    saturation_weight=0.1,
    seed=42,
):
    """Train ColorizeNet using streaming data from data/data_preprocess.py."""
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    if checkpoint_dir is None:
        checkpoint_dir = Path(__file__).resolve().parent
    else:
        checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    repo_root = Path(__file__).resolve().parents[2]
    if str(repo_root) not in os.sys.path:
        os.sys.path.insert(0, str(repo_root))

    from data.colorizing_data_preprocess import build_colorization_dataloader

    dataloader = build_colorization_dataloader(
        batch_size=batch_size,
        image_size=(image_size, image_size),
        num_workers=num_workers,
        seed=seed,
    )

    model = ColorizeNet().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.L1Loss()

    def combined_loss(ab_pred, ab_target, saturation_weight=0.1):
        """L1 loss + saturation penalty to encourage colorful predictions."""
        l1 = criterion(ab_pred, ab_target)
        
        # Saturation loss: penalize low magnitude in ab space
        # ||ab|| should be high for colorful outputs
        ab_magnitude = torch.sqrt(torch.sum(ab_pred ** 2, dim=1, keepdim=True) + 1e-8)
        saturation_loss = torch.mean(torch.clamp(0.5 - ab_magnitude, min=0))
        
        return l1 + saturation_weight * saturation_loss

    model.train()
    for epoch in range(1, epochs + 1):
        running_loss = 0.0
        step = 0

        data_iter = iter(dataloader)
        while step < steps_per_epoch:
            try:
                l_input, ab_target = next(data_iter)
            except StopIteration:
                data_iter = iter(dataloader)
                l_input, ab_target = next(data_iter)

            non_blocking = device != "cpu"
            l_input = l_input.to(device, non_blocking=non_blocking)
            ab_target = ab_target.to(device, non_blocking=non_blocking)

            optimizer.zero_grad(set_to_none=True)
            ab_pred = model(l_input)
            loss = combined_loss(ab_pred, ab_target, saturation_weight=saturation_weight)
            loss.backward()
            optimizer.step()

            running_loss += float(loss.item())
            step += 1

        avg_loss = running_loss / float(steps_per_epoch)
        print(f"Epoch [{epoch}/{epochs}] - loss: {avg_loss:.6f}")

        epoch_ckpt = checkpoint_dir / f"co_model_1_epoch_{epoch}.pth"
        torch.save(model.state_dict(), epoch_ckpt)

    final_ckpt = checkpoint_dir / checkpoint_name
    torch.save(model.state_dict(), final_ckpt)
    print(f"Training complete. Saved final checkpoint to: {final_ckpt}")
    return str(final_ckpt)


def _build_arg_parser():
    parser = argparse.ArgumentParser(description="Train co_model_1 colorization model")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--steps-per-epoch", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--checkpoint-dir", type=str, default=None)
    parser.add_argument("--checkpoint-name", type=str, default="co_model_1.pth")
    parser.add_argument("--log-interval", type=int, default=10)
    parser.add_argument("--saturation-weight", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    return parser


if __name__ == "__main__":
    args = _build_arg_parser().parse_args()
    train_colorize_net(
        epochs=args.epochs,
        steps_per_epoch=args.steps_per_epoch,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        image_size=args.image_size,
        num_workers=args.num_workers,
        device=args.device,
        checkpoint_dir=args.checkpoint_dir,
        checkpoint_name=args.checkpoint_name,
        log_interval=args.log_interval,
        saturation_weight=args.saturation_weight,
        seed=args.seed,
    )

"""
Training:

Download color image from COCO stream
Convert RGB → Lab color space (this splits color into 3 channels: L=lightness, a=green-red, b=blue-yellow)
Extract L channel only → feed into model
Model outputs predicted ab channels
Compare predicted ab vs. ground-truth ab (from original image) → compute L1 loss → backprop

Inference (colorization):

Take grayscale image (or extract its L channel from Lab)
Feed L into model → get predicted ab
Merge L + predicted ab → convert back to RGB → colorized image
So yes: L (lightness/grayscale) in → model → ab (color) out.

(--epochs 20 --steps-per-epoch 100 --batch-size 4)
20 epochs × 100 steps × 4 images = 8,000 images processed during training, which should be enough to see some meaningful colorization results.
"""