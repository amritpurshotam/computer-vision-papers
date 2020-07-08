import tensorflow_datasets as tfds

ds, info = tfds.load(
    "imagenette/320px-v2",
    split=["train", "validation"],
    data_dir="F:\\imagenette",
    with_info=True,
)

fig = tfds.show_examples(ds, info)
