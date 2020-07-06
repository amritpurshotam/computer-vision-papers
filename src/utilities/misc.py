import matplotlib.pyplot as plt


def get_mappings() -> dict:
    mappings = {}
    f = open("F:\\caffe_ilsvrc12.tar\\caffe_ilsvrc12\\synset_words.txt")
    for line in f:
        splits = line.split(" ", 1)
        folder_name = splits[0]
        class_names = splits[1]
        class_name = class_names.split(",", 1)[0]
        class_name = class_name.rstrip()
        mappings[folder_name] = class_name
    f.close()
    return mappings


def show_batch(image_batch, label_batch, class_names):
    mappings = get_mappings()
    plt.figure(figsize=(10, 10))
    for n in range(25):
        plt.subplot(5, 5, n + 1)
        plt.imshow(image_batch[n])
        name = mappings[class_names[label_batch[n] == 1][0]]
        plt.title(name)
        plt.axis("off")
    plt.show()
