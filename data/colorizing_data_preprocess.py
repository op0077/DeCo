from datasets import interleave_datasets, load_dataset
import cv2
import numpy as np
import torch
import warnings
from urllib.request import urlopen
from torch.utils.data import DataLoader, IterableDataset, get_worker_info


def build_streaming_dataset(seed=42, probabilities=None):
    """Build a mixed streaming dataset from COCO + DIV2K."""
    if probabilities is None:
        probabilities = [0.6, 0.4]

    coco = load_dataset("nickpai/coco2017-colorization", streaming=True)

    try:
        div2k = load_dataset("eugenesiow/Div2k", streaming=True)
    except Exception as exc:
        warnings.warn(
            f"DIV2K streaming unavailable ({exc}). Falling back to COCO-only stream."
        )
        return coco["train"]

    return interleave_datasets(
        [coco["train"], div2k["train"]],
        probabilities=probabilities,
        seed=seed,
    )


def _extract_rgb_image(sample):
    """Return an RGB uint8 ndarray from a dataset sample."""
    image = sample.get("image")

    if image is None:
        image_url = sample.get("coco_url")
        if not image_url:
            return None

        try:
            with urlopen(image_url, timeout=10) as response:
                encoded = np.asarray(bytearray(response.read()), dtype=np.uint8)
            bgr = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
            if bgr is None:
                return None
            image = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        except Exception:
            return None

    if hasattr(image, "convert"):
        image = np.array(image.convert("RGB"), dtype=np.uint8)
    else:
        image = np.array(image, dtype=np.uint8)

    if image.ndim == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    elif image.ndim == 3 and image.shape[2] == 4:
        image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)

    if image.ndim != 3 or image.shape[2] != 3:
        return None

    return image


def preprocess_for_colorization(rgb_image, image_size=(256, 256)):
    """
    Convert RGB image into model tensors.

    Returns:
      l_tensor: [1, H, W] normalized to [-1, 1]
      ab_tensor: [2, H, W] normalized to [-1, 1]
    """
    resized = cv2.resize(rgb_image, image_size)
    lab = cv2.cvtColor(resized, cv2.COLOR_RGB2LAB)

    l_channel = lab[:, :, 0].astype(np.float32)
    ab_channels = lab[:, :, 1:].astype(np.float32)

    l_tensor = torch.from_numpy((l_channel / 127.5) - 1.0).unsqueeze(0)
    ab_tensor = torch.from_numpy((ab_channels / 127.5) - 1.0).permute(2, 0, 1)
    return l_tensor, ab_tensor


class StreamingColorizationDataset(IterableDataset):
    """Iterable dataset that yields (L, ab) tensors from streaming image sources."""

    def __init__(self, image_size=(256, 256), seed=42, probabilities=None, max_samples=None):
        super().__init__()
        self.image_size = image_size
        self.seed = seed
        self.probabilities = probabilities
        self.max_samples = max_samples

    def __iter__(self):
        stream = build_streaming_dataset(seed=self.seed, probabilities=self.probabilities)

        worker_info = get_worker_info()
        if worker_info is not None and hasattr(stream, "shard"):
            stream = stream.shard(num_shards=worker_info.num_workers, index=worker_info.id)

        count = 0
        for sample in stream:
            if self.max_samples is not None and count >= self.max_samples:
                break

            rgb = _extract_rgb_image(sample)
            if rgb is None:
                continue

            try:
                l_tensor, ab_tensor = preprocess_for_colorization(rgb, image_size=self.image_size)
                yield l_tensor, ab_tensor
                count += 1
            except Exception:
                continue


def build_colorization_dataloader(
    batch_size=16,
    image_size=(256, 256),
    num_workers=0,
    seed=42,
    probabilities=None,
    max_samples=None,
):
    """Create a dataloader for streaming colorization training data."""
    dataset = StreamingColorizationDataset(
        image_size=image_size,
        seed=seed,
        probabilities=probabilities,
        max_samples=max_samples,
    )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )