# This is a sample Python script.

# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.
import torch
import torchvision
from model import *


# Press the green button in the gutter to run the script.
if __name__ == '__main__':

    print('AdvCNN')

    train_dataloader, test_dataloader, device = init_data()
    loss_fn = nn.CrossEntropyLoss()

    model = CNN()
    model.load_state_dict(torch.load("distillation_teacher_CNN.pth"))
    print("Load Pytorch Model")

    test(test_dataloader, model, loss_fn, device)

