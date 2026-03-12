from datasets import load_dataset, interleave_datasets
import webdataset as wds

# 1. ImageNet via WebDataset streaming dataset
imagenet = (
    wds.WebDataset("https://huggingface.co/datasets/imagenet-1k/resolve/main/data/train-00000-of-01024.tar")
    .decode("rgb")
    .to_tuple("jpg;png;jpeg", "cls")
)

# 2. COCO via HuggingFace streaming dataset
coco = load_dataset("nickpai/coco2017-colorization", streaming=True)

# 3. DIV2K streaming dataset
div2k = load_dataset("eugenesiow/Div2k", streaming=True)

# Mix all datasets
combined = interleave_datasets(
    [coco["train"], div2k["train"]],
    probabilities=[0.6, 0.4],  # weight COCO more since it's larger
    seed=42
)

for sample in combined:
    image = sample["image"]

    # TODO:feed into your preprocessing pipeline