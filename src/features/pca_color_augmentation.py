import matplotlib.pyplot as plt
import numpy as np
from skimage import io


def fancy_pca(original_image):
    """PCA Color Augmentation

    Code taken from https://medium.com/@adityaraj_64455/it-all-started-with-cnns-alexnet-3023b21bb891
    """
    # first you need to unroll the image into a nx3 where 3 is the no. of colour channels

    print(original_image.shape)
    renorm_image = np.reshape(
        original_image, (np.dot(original_image.shape[0], original_image.shape[1]), 3)
    )

    # Before applying PCA you must normalize the data in each column separately as we will be applying PCA column-wise

    mean = np.mean(renorm_image, axis=0)  # computing the mean
    std = np.std(renorm_image, axis=0)  # computing the standard deviation
    # we change the datatpe so as to avoid any warnings or errors
    renorm_image = renorm_image.astype("float32")
    renorm_image -= mean
    renorm_image /= std
    # next we normalize the data using the 2 columns

    cov = np.cov(renorm_image, rowvar=False)
    # finding the co-variance matrix for computing the eigen values
    # and eigen vectors.

    lambdas, p = np.linalg.eig(cov)
    # finding the eigen values lambdas and the vectors p
    # of the covarince matrix

    alphas = np.random.normal(0, 0.1, 3)
    # aplha here is the gaussian random no. generated

    delta = np.dot(p, alphas * lambdas)
    # delta here represents the value which will be
    # added to the re_norm image

    pca_augmentation_version_renorm_image = renorm_image + delta
    # forming augmented normalised image

    pca_color_image = pca_augmentation_version_renorm_image * std + mean
    # de-normalising the image

    pca_color_image = np.maximum(np.minimum(pca_color_image, 255), 0).astype("uint8")
    # necessary conditions which need to be checked

    pca_color_image = np.ravel(pca_color_image).reshape(
        (original_image.shape[0], original_image.shape[1], 3)
    )
    # rollong back the image into a displayable just as
    # original image
    return pca_color_image


if __name__ == "__main__":
    img = io.imread("cat.jpeg")

    fig = plt.figure(figsize=(8, 8))
    columns = 4
    rows = 5

    fig.add_subplot(rows, columns, 1)
    plt.imshow(img)
    for i in range(2, columns * rows + 1):
        pca = fancy_pca(img)
        fig.add_subplot(rows, columns, i)
        plt.imshow(pca)
    plt.show()
