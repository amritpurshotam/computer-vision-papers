import glob
import os
import re
import shutil
from typing import Tuple

from tensorflow.keras.callbacks import Callback
from tensorflow.keras.models import Model, load_model


class LastModelManager(Callback):
    def __init__(self, base_dir: str):
        self.base_dir = base_dir
        super().__init__()

    def on_epoch_end(self, epoch, logs=None):
        if not os.path.exists(self.base_dir):
            os.mkdir(self.base_dir)

        existing_models = os.listdir(self.base_dir)
        for existing_model in existing_models:
            if existing_model.startswith(f"last_model_{epoch:02d}"):
                print(f"\nDeleting previous model {existing_model}...")
                shutil.rmtree(os.path.join(self.base_dir, existing_model))
                break
        return super().on_epoch_end(epoch, logs=logs)


def get_epoch_from_last_model_path(last_model_path: str) -> int:
    match = re.search(r"last_model_(\d{2})", last_model_path)
    if match is not None:
        epoch = int(match.group(1))
        return epoch
    return 0


def get_last_model_path(base_dir: str):
    paths = glob.glob(os.path.join(base_dir, "last_model_*"))
    if paths:
        last_model_path = paths[0]
        return last_model_path
    return None


def get_latest_trained_model(model: Model, base_dir: str) -> Tuple[Model, int]:
    initial_epoch = 0
    last_model_path = get_last_model_path(base_dir)
    if last_model_path:
        initial_epoch = get_epoch_from_last_model_path(last_model_path)
        model = load_model(last_model_path)
    return model, initial_epoch
