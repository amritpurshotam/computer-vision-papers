import click
import wandb
from tensorflow.keras.callbacks import ReduceLROnPlateau
from wandb.keras import WandbCallback

from src.callbacks.helper import get_best_model_checkpoint
from src.config import get_dataset_config
from src.models.alexnet.dataset_loader import DatasetLoader
from src.models.alexnet.hyperparams import EPOCHS, LR_DECAY, MIN_LR
from src.models.alexnet.model import get_alexnet_model


@click.command()
@click.option(
    "--dataset", type=click.Choice(["imagenet", "imagenette"], case_sensitive=False)
)
@click.option(
    "--group", type=str, required=True, help="What to organise this experiment under"
)
@click.option(
    "--name", type=str, required=True, help="The name of this specific experiment"
)
@click.option(
    "--apply_pca",
    type=bool,
    required=False,
    default=True,
    help="Whether or not to apply PCA Augmentation",
)
@click.option(
    "--imagenet_pca",
    type=bool,
    required=False,
    default=False,
    help=(
        "Whether or not to use pre-computed principal components from the "
        + "whole of ImageNet"
    ),
)
def train(dataset: str, group: str, name: str, apply_pca: bool, imagenet_pca: bool):
    wandb.init(project="computer-vision-papers", group=group, name=name, tags=[dataset])

    ds_config = get_dataset_config(dataset)
    train_dir = ds_config["train"]["path"]
    val_dir = ds_config["val"]["path"]
    train_samples = ds_config["train"]["samples"]
    val_samples = ds_config["val"]["samples"]

    preprocessor = DatasetLoader(
        train_dir, train_samples, val_dir, val_samples, apply_pca, imagenet_pca
    )

    train_ds = preprocessor.build_train_ds()
    val_ds = preprocessor.build_train_ds()

    base_dir = f".\\models\\alexnet\\{dataset}\\{group}\\{name}"
    best_model_checkpoint = get_best_model_checkpoint(base_dir)

    model = get_alexnet_model(preprocessor.num_classes)

    scheduler = ReduceLROnPlateau(
        monitor="val_accuracy",
        factor=LR_DECAY,
        mode="max",
        patience=8,
        min_lr=MIN_LR,
        min_delta=0.0001,
    )

    model.fit(
        train_ds,
        epochs=EPOCHS,
        validation_data=val_ds,
        callbacks=[
            scheduler,
            best_model_checkpoint,
            WandbCallback(
                save_model=False,
                log_weights=True,
                log_gradients=True,
                training_data=train_ds,
            ),
        ],
    )

    model.save(f"{base_dir}_model")


if __name__ == "__main__":
    train()
