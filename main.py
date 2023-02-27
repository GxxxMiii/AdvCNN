# This is a sample Python script.

# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.
import torch
import torchvision
import foolbox as fb
from torchvision.models import ResNet18_Weights


def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press ⌘F8 to toggle the breakpoint.


# Press the green button in the gutter to run the script.
if __name__ == '__main__':

    model = torchvision.models.resnet18(weights=ResNet18_Weights.DEFAULT).eval()
    processing = dict(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], axis=-3)
    bounds = (0, 1)
    fmodel = fb.PyTorchModel(model, bounds=bounds)

    print(fmodel.bounds)

