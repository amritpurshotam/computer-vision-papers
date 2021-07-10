import os

from tensorflow.keras.callbacks import ModelCheckpoint


def get_last_model_checkpoint(base_dir: str) -> ModelCheckpoint:
    last_model_checkpoint = ModelCheckpoint(
        filepath=os.path.join(base_dir, "last_model"),
        save_best_only=False,
        save_weights_only=False,
        save_freq="epoch",
    )
    return last_model_checkpoint


def get_best_model_checkpoint(base_dir: str) -> ModelCheckpoint:
    best_model_checkpoint = ModelCheckpoint(
        filepath=os.path.join(base_dir, "best_model"),
        save_best_only=True,
        save_weights_only=False,
        monitor="val_accuracy",
        mode="max",
    )
    return best_model_checkpoint
