import torch
import torchvision
import torch.nn.functional as F
import numpy as np
from model import *


def eva_defense(clean_cnn, defended_cnn):

    clean_params = torch.cat([p.flatten() for p in clean_cnn.parameters()])
    defended_params = torch.cat([p.flatten() for p in defended_cnn.parameters()])

    params_combined = torch.cat([clean_params, defended_params])
    probs_combined = F.softmax(params_combined, dim=0)

    num_params = len(clean_params)
    kl_div = F.kl_div(torch.log(probs_combined[:num_params]), probs_combined[num_params:], reduction='sum')
    print(f"parameters distortion: {kl_div.item():.10f}")


if __name__ == '__main__':

    clean_cnn = CNN()
    clean_cnn.load_state_dict(torch.load("models/clean_CNN.pth"))

    distilled_cnn = CNN()
    distilled_cnn.load_state_dict(torch.load("models/regularized_L100_CNN.pth"))

    print("Regularized CNN:")
    eva_defense(clean_cnn, distilled_cnn)

