from pathlib import Path

import numpy as np
from PIL import Image

from src.config import get_dataset_config


def process(image_path: Path) -> np.ndarray:
    image = Image.open(image_path, formats=["JPEG"])
    if image.mode == "CMYK" or image.mode == "L":
        image = image.convert("RGB")
    elif image.mode != "RGB":
        print(f"{image.mode}: {image_path}")
    image = np.asarray(image)
    image = image.astype(np.float32)
    image = image / 255.0
    return image


if __name__ == "__main__":
    ds_config = get_dataset_config("imagenet")
    train_dir = ds_config["train"]["path"]
    num_images = ds_config["train"]["samples"]

    image_paths = train_dir.glob("*/*")
    means = np.zeros((num_images, 3))
    stds = np.zeros((num_images, 3))
    i = 0
    for image_path in iter(image_paths):
        try:
            image = process(image_path)
            mean = np.mean(image, axis=(0, 1))
            std = np.std(image, axis=(0, 1))
            means[i] = mean
            stds[i] = std
            i = i + 1
        except Exception as e:
            print(image_path)
            print(e)
    ds_means = np.mean(means, axis=0)
    ds_stds = np.mean(stds, axis=0)

    print(ds_means.shape)
    print(ds_stds.shape)

    print("Dataset means")
    print(ds_means)
    print("Dataset standard deviations")
    print(ds_stds)
