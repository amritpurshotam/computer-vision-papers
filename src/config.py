import json
import pathlib


def get_dataset_config(dataset: str):
    with open("config.json", "r") as f:
        config = json.load(f)

    if dataset == "imagenet":
        ds_config = config[dataset]
    elif dataset == "imagenette":
        ds_config = config[dataset]
    else:
        raise ValueError("Invalid dataset")

    ds_config["train"]["path"] = pathlib.Path(ds_config["train"]["path"])  # type: ignore # noqa: B950
    ds_config["val"]["path"] = pathlib.Path(ds_config["val"]["path"])  # type: ignore
    return ds_config
