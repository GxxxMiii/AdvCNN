import foolbox
import numpy as np
import torch
import random
import matplotlib.pyplot as plt
from matplotlib import use as mpl_use
from foolbox.models import PyTorchModel
from foolbox.attacks import DeepFoolAttack
from foolbox.criteria import Misclassification
from model import CNN


if __name__ == '__main__':

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using {device} device")
    cnn = CNN().to(device)

    # load model
    clean_cnn = CNN().eval()
    # print(clean_cnn)
    clean_cnn.load_state_dict(torch.load("clean_CNN.pth"))
    print("Load Pytorch Model from clean_CNN.pth\n")

    # turn into foolbox model
    fmodel = PyTorchModel(model=clean_cnn, bounds=(0, 1), num_classes=10)

    # Deepfool attack
    criterion = Misclassification()
    attack = DeepFoolAttack(model=fmodel, criterion=criterion)

    image, label = foolbox.utils.samples(dataset='mnist', batchsize=1, index=random.randint(0, 20), bounds=(0, 1))
    image = image[np.newaxis, :]
    print('true label: ', label)
    pre_label = clean_cnn(torch.tensor(image).to(device))
    print('prediction label: ', np.argmax(pre_label.detach().numpy()))
    adversarial = attack(inputs=image, labels=label)
    adv_label = clean_cnn(torch.tensor(adversarial).to(device))
    torch.set_printoptions(precision=4, sci_mode=False)
    softmax = torch.nn.Softmax(dim=1)
    print(softmax(adv_label).detach())
    print('adv prediction label: ', np.argmax(adv_label.detach().numpy()))

    # plot the example
    mpl_use('MacOSX')

    plt.subplot(1, 3, 1)
    image = image.reshape(28, 28)
    plt.imshow(image, cmap='gray')

    plt.subplot(1, 3, 2)
    perturbation = adversarial - image
    perturbation = perturbation.reshape(28, 28)
    plt.imshow(perturbation, cmap='gray')

    plt.subplot(1, 3, 3)
    adversarial = adversarial.reshape(28, 28)
    plt.imshow(adversarial, cmap='gray')

    plt.savefig('pics/Deepfool_adv.png')

    print("Done!")
