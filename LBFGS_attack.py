import foolbox
import numpy as np
import torch
import random
import matplotlib.pyplot as plt
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
from foolbox.models import PyTorchModel
from foolbox import zoo
from foolbox.v1.attacks import LBFGSAttack
from foolbox.criteria import TargetClassProbability
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
    # fmodel = PyTorchModel(model=clean_cnn, bounds=(0, 255), num_classes=10)
    fmodel = zoo.get_model(url="https://github.com/bethgelab/AnalysisBySynthesis")

    # L-BFGS attack
    target_class = 3
    criterion = TargetClassProbability(target_class=target_class, p=0.9)
    attack = LBFGSAttack(model=fmodel, criterion=criterion)

    image, label = foolbox.utils.samples(dataset='mnist', batchsize=1, index=random.randint(0, 10000), bounds=(0, 255))
    print('true label: ', label)
    pre_label = clean_cnn.forward(torch.tensor(image[np.newaxis, :]).to(device))
    print('prediction label: ', np.argmax(pre_label.detach().numpy()))
    # print(image, label)
    adversarial = attack(input_or_adv=image, label=label)

    # # plot the example
    # plt.subplot(1, 3, 1)
    # plt.imshow(image)
    #
    # plt.subplot(1, 3, 2)
    # plt.imshow(adversarial)
    #
    # plt.subplot(1, 3, 3)
    # plt.imshow(adversarial - image)

    print("Done!")
