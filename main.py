# This is a sample Python script.

# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.
import torch
import torchvision
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import use as mpl_use
from model import *


# Press the green button in the gutter to run the script.
if __name__ == '__main__':

    print('AdvCNN')

    train_dataloader, test_dataloader, device = init_data()
    loss_fn = nn.CrossEntropyLoss()

    clean_cnn = CNN()
    clean_cnn.load_state_dict(torch.load("models/clean_CNN.pth"))

    distilled_cnn = CNN()
    distilled_cnn.load_state_dict(torch.load("models/distilled_CNN.pth"))

    regularized_cnn = CNN()
    regularized_cnn.load_state_dict(torch.load("models/regularized_CNN.pth"))
    print("Load Pytorch Model")

    clean_params = list(clean_cnn.parameters())
    distilled_params = list(distilled_cnn.parameters())
    regularized_params = list(distilled_cnn.parameters())

    dist = 0
    for i in range(len(clean_params)):
        dist += np.linalg.norm(clean_params[i].detach().numpy() - distilled_params[i].detach().numpy())
    print(dist)

    # test(test_dataloader, clean_cnn, loss_fn, device)
    # test(test_dataloader, distilled_cnn, loss_fn, device)
    # test(test_dataloader, regularized_cnn, loss_fn, device)

    # show gradients for different digits
    mpl_use('MacOSX')


