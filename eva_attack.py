import time

import foolbox
import numpy as np
import random
import matplotlib.pyplot as plt
from torch.utils.data import RandomSampler, DataLoader
from matplotlib import use as mpl_use
from attacks import *
from model import *


def init_eva(d):
    """

    :param d: num of samples
    :return: evaluation dataloader
    """

    test_data = datasets.MNIST(root="data", train=False, download=True, transform=ToTensor(), )
    sampler = RandomSampler(data_source=test_data, replacement=False, num_samples=d)
    eva_dataloader = DataLoader(dataset=test_data, batch_size=1, sampler=sampler)

    # Get cpu or gpu device for training.
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using {device} device")

    return eva_dataloader, device


def eva_attack(model, attack_flag):
    """

    :param model: pytorch model
    :param attack_flag: 0 for LBFGS, 1 for FGSM, 2 for JSMA, 3 for Deepfool
    """

    torch.set_printoptions(precision=4, sci_mode=False)

    d = 1000
    success = 0
    confidence = 0
    cost = 0
    omega = 0.4
    fgsm_epsilon = 0.3
    eva_dataloader, device = init_eva(d)

    for batch, (X, y) in enumerate(eva_dataloader):
        start_time = time.time()
        if attack_flag == 0:
            adv = lbfgs_attack(model=model, image=X[0], label=y[0])
        elif attack_flag == 1:
            adv = fgsm_attack(model=model, image=X[0], label=y[0], epsilon=fgsm_epsilon)
        elif attack_flag == 2:
            adv = jsma_attack(model=model, image=X[0], label=y[0])
        elif attack_flag == 3:
            adv = deepfool_attack(model=model, image=X[0], label=y[0])
        else:
            adv = X[0]
        run_time = time.time() - start_time
        if adv is not None:
            rho = adv - X
            rho_p = np.linalg.norm(rho) / np.linalg.norm(X)
            if rho_p < omega:
                success += 1

                adv_label = model(adv.reshape(1, 1, 28, 28).to(device))
                softmax = torch.nn.Softmax(dim=1)
                confidence += torch.max(softmax(adv_label.detach()))

                cost += run_time

        if (batch+1) % 100 == 0:
            print(f"success ratio: [{success:>4d}/{d:>4d}]", end='  ')
            print(f"average confidence: {confidence/success:.4f}")
            print(f"average computation cost: {cost/success:.4f}")

    print()
    print(f"success ratio: ", success/d)
    print(f"average confidence: {confidence / success:.4f}")
    print(f"average computation cost: {cost / success:.4f}")

    return success/d, confidence / success


if __name__ == '__main__':

    # load model
    cnn = CNN().eval()
    # print(clean_cnn)
    cnn.load_state_dict(torch.load("models/regularized_CNN.pth"))
    print("Load Pytorch Model from regularized_CNN.pth\n")

    eva_attack(model=cnn, attack_flag=1)
