import json
import pathlib

with open("config.json", "r") as f:
    config = json.load(f)


def get_dataset_config(dataset: str):

    if dataset == "imagenet":
        ds_config = config[dataset]
    elif dataset == "imagenette":
        ds_config = config[dataset]
    else:
        raise ValueError("Invalid dataset type")

    ds_config["train"]["path"] = pathlib.Path(ds_config["train"]["path"])  # type: ignore # noqa: B950
    ds_config["val"]["path"] = pathlib.Path(ds_config["val"]["path"])  # type: ignore
    return ds_config
