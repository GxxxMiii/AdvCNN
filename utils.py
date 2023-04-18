import time
import torch
import matplotlib.pyplot as plt
from matplotlib import use as mpl_use


def save_pics(dir, image, adversarial):
    """

    :param dir: directory to save (str)
    :param image: clean example
    :param adversarial: adversarial example
    :return:
    """

    plt.subplot(1, 3, 1)
    image_plt = image.reshape(28, 28)
    plt.imshow(image_plt, cmap='gray')

    plt.subplot(1, 3, 2)
    perturbation = adversarial - image
    perturbation = perturbation.reshape(28, 28)
    plt.imshow(perturbation, cmap='gray')

    plt.subplot(1, 3, 3)
    adversarial = adversarial.reshape(28, 28)
    plt.imshow(adversarial, cmap='gray')

    name = time.time_ns()
    plt.savefig(dir + '/adv-' + str(name) + '.png')

