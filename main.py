# This is a sample Python script.

# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.
import torch
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
    clean_cnn.to(device)

    distilled_cnn = CNN()
    distilled_cnn.load_state_dict(torch.load("models/distilled_CNN.pth"))
    distilled_cnn.to(device)

    print("Load Pytorch Model")

    test(test_dataloader, clean_cnn, loss_fn, device)
    test(test_dataloader, distilled_cnn, loss_fn, device)



