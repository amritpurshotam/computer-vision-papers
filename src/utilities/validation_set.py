import os
import shutil


def sort_validation_images():
    num_to_class = {}
    synsets = open("F:\\caffe_ilsvrc12.tar\\caffe_ilsvrc12\\synsets.txt")
    i = 0
    for line in synsets:
        num_to_class[i] = line.rstrip()
        i = i + 1
    synsets.close()

    val_path = "F:\\ILSVRC2012_img_val"
    val_mappings = open("F:\\caffe_ilsvrc12.tar\\caffe_ilsvrc12\\val.txt")
    for line in val_mappings:
        image_name, num = line.split(" ")
        num = int(num)
        class_name = num_to_class[num]

        source_image_path = os.path.join(val_path, image_name)
        destination_directory = os.path.join(val_path, class_name)
        destination_image_path = os.path.join(destination_directory, image_name)

        if not os.path.isdir(destination_directory):
            os.mkdir(destination_directory)
        shutil.move(source_image_path, destination_image_path)
    val_mappings.close()


if __name__ == "__main__":
    sort_validation_images()
