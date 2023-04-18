import time

import foolbox
import numpy as np
import random
import matplotlib.pyplot as plt
import torch
from matplotlib import use as mpl_use
from foolbox.models import PyTorchModel
from foolbox.v1.attacks import GradientSignAttack
from foolbox.v1.attacks import LBFGSAttack
from foolbox.v1.attacks import SaliencyMapAttack
from foolbox.v1.attacks import DeepFoolAttack
from foolbox.criteria import TargetClass
from foolbox.criteria import Misclassification
from model import *


def lbfgs_attack(model, image, label):
    """

    :param model: pytorch model
    :param image: clean example as tensor
    :param label: label as tensor
    :return: LBFGS adversarial example as tensor
    """
    # turn into foolbox model
    fmodel = PyTorchModel(model=model, bounds=(0, 1), num_classes=10)

    # L-BFGS attack
    target_class = random.randint(0, 9)
    while target_class == int(label):
        target_class = random.randint(0, 9)
    criterion = TargetClass(target_class=target_class)
    attack = LBFGSAttack(model=fmodel, criterion=criterion)

    adversarial = attack(input_or_adv=image.numpy(), label=label.numpy())
    if adversarial is not None:
        lbfgs_adv = torch.tensor(adversarial)
    else:
        lbfgs_adv = adversarial
    return lbfgs_adv


def fgsm_attack(model, image, label, epsilon):
    """

    :param model: pytorch model
    :param image: clean example as tensor
    :param label: label as tensor
    :param epsilon: perturbation epsilon
    :return: FGSM adversarial example as tensor
    """
    # turn into foolbox model
    fmodel = PyTorchModel(model=model, bounds=(0, 1), num_classes=10)

    # FGSM attack
    criterion = Misclassification()
    attack = GradientSignAttack(model=fmodel, criterion=criterion)

    adversarial = attack(input_or_adv=image.numpy(), label=label.numpy(), epsilons=1, max_epsilon=epsilon)

    if adversarial is not None:
        fgsm_adv = torch.tensor(adversarial)
    else:
        fgsm_adv = adversarial
    return fgsm_adv


def jsma_attack(model, image, label):
    """

    :param model: pytorch model
    :param image: clean example as tensor
    :param label: label as tensor
    :return: JSMA adversarial example as tensor
    """
    # turn into foolbox model
    fmodel = PyTorchModel(model=model, bounds=(0, 1), num_classes=10)

    # JSMA attack
    target_class = random.randint(0, 9)
    while target_class == int(label):
        target_class = random.randint(0, 9)
    criterion = TargetClass(target_class=target_class)
    attack = SaliencyMapAttack(model=fmodel, criterion=criterion)

    adversarial = attack(input_or_adv=image.numpy(), label=label.numpy())
    if adversarial is not None:
        jsma_adv = torch.tensor(adversarial)
    else:
        jsma_adv = adversarial
    return jsma_adv


def deepfool_attack(model, image, label):
    """

    :param model: pytorch model
    :param image: clean example as tensor
    :param label: label as tensor
    :return: JSMA adversarial example as tensor
    """
    # turn into foolbox model
    fmodel = PyTorchModel(model=model, bounds=(0, 1), num_classes=10)

    # Deepfool attack
    criterion = Misclassification()
    attack = DeepFoolAttack(model=fmodel, criterion=criterion)

    adversarial = attack(input_or_adv=image.numpy(), label=label.numpy())

    if adversarial is not None:
        deepfool_adv = torch.tensor(adversarial)
    else:
        deepfool_adv = adversarial
    return deepfool_adv


if __name__ == '__main__':

    train_dataloader, test_dataloader, device = init_data()

    # load model
    clean_cnn = CNN().eval()
    # print(clean_cnn)
    clean_cnn.load_state_dict(torch.load("models/clean_CNN.pth"))
    print("Load Pytorch Model from clean_CNN.pth\n")

    epsilon = 0.2

    (test_image, test_label) = test_dataloader.dataset[random.randint(0, 9999)]
    pre_label = clean_cnn(test_image.reshape(1, 1, 28, 28).to(device))
    softmax = torch.nn.Softmax(dim=1)
    torch.set_printoptions(precision=4, sci_mode=False)
    print(softmax(pre_label.detach()))
    print('prediction label: ', np.argmax(pre_label.detach().numpy()))

    start_time = time.time()
    # adv = fgsm_attack(model=clean_cnn, image=test_image, label=torch.tensor(test_label), epsilon=epsilon)
    # adv = lbfgs_attack(model=clean_cnn, image=test_image, label=torch.tensor(test_label))
    # adv = jsma_attack(model=clean_cnn, image=test_image, label=torch.tensor(test_label))
    adv = deepfool_attack(model=clean_cnn, image=test_image, label=torch.tensor(test_label))
    run_time = time.time() - start_time
    adv_label = clean_cnn(adv.reshape(1, 1, 28, 28).to(device))
    torch.set_printoptions(precision=4, sci_mode=False)
    print(softmax(adv_label.detach()))
    print(torch.max(softmax(adv_label.detach())))
    print('adv prediction label: ', np.argmax(adv_label.detach().numpy()))
    print(f"run time: {run_time:.4f}")

    # plot the example
    mpl_use('MacOSX')

    plt.subplot(1, 3, 1)
    image_plt = test_image.reshape(28, 28)
    plt.imshow(image_plt, cmap='gray')

    plt.subplot(1, 3, 2)
    perturbation = adv - test_image
    dist = np.linalg.norm(perturbation)
    image_d = np.linalg.norm(test_image)
    print(f"dist: {dist:.4f}  image_dist: {image_d:.4f}  p: {dist/image_d:.4f}")
    perturbation = perturbation.reshape(28, 28)
    plt.imshow(perturbation, cmap='gray')

    plt.subplot(1, 3, 3)
    adversarial = adv.reshape(28, 28)
    plt.imshow(adversarial, cmap='gray')

    plt.savefig('pics/adv.png')

    print("Done!")
