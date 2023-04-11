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
    clean_cnn.load_state_dict(torch.load("clean_CNN.pth"))

    distilled_cnn = CNN()
    distilled_cnn.load_state_dict(torch.load("distilled_CNN.pth"))
    for name, param in distilled_cnn.state_dict().items():
        print(name)
        print(type(param), param.size())

    # dist_params = np.linalg.norm(distilled_cnn.parameters())
    # clean_params = np.linalg.norm(clean_cnn.parameters())
    # print('dist_params:', dist_params)
    # print('clean_params:', clean_params)

    # regularized_cnn = CNN()
    # regularized_cnn.load_state_dict(torch.load("regularized_CNN.pth"))
    # print("Load Pytorch Model")

    # test(test_dataloader, clean_cnn, loss_fn, device)
    # test(test_dataloader, distilled_cnn, loss_fn, device)
    # test(test_dataloader, regularized_cnn, loss_fn, device)

    # show gradients for different digits
    mpl_use('MacOSX')


