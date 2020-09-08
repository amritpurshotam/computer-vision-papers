import os
import shutil

from tensorflow.keras.callbacks import Callback


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
