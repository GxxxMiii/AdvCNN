import random
import numpy as np
from model import *
import torch
from attacks import *


def adv_train(dataloader, model, loss_fn, optimizer, device, attack_flag, epsilon=0.2):
    """

    :param attack_flag: 0 for LBFGS, 1 for FGSM, 2 for JSMA, 3 for Deepfool
    :param epsilon: epsilon for FGSM (default=0.2)
    :return:
    """
    size = len(dataloader.dataset)
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        # adversarial training
        if batch > 10 and batch % 2 == 0:
            for i in range(len(X)):
                if attack_flag == 0:
                    adv = lbfgs_attack(model=model.eval(), image=X[0], label=y[0])
                elif attack_flag == 1:
                    adv = fgsm_attack(model=model.eval(), image=X[0], label=y[0], epsilon=epsilon)
                elif attack_flag == 2:
                    adv = jsma_attack(model=model.eval(), image=X[0], label=y[0])
                elif attack_flag == 3:
                    adv = deepfool_attack(model=model.eval(), image=X[0], label=y[0])
                else:
                    adv = X[0]
                if adv is not None:
                    X[i] = adv

        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            loss, current = loss.item(), (batch + 1) * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


if __name__ == '__main__':

    learning_rate = 0.0002
    train_dataloader, test_dataloader, device = init_data()
    adv_cnn = CNN().to(device)
    # lenet = LeNet().to(device)

    attack_flag = 1
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(adv_cnn.parameters(), lr=learning_rate)

    epochs = 1
    for t in range(epochs):
        print(f"Epoch {t + 1}\n-------------------------------")
        adv_train(train_dataloader, adv_cnn, loss_fn, optimizer, device, attack_flag=attack_flag, epsilon=0.3)
        test(test_dataloader, adv_cnn, loss_fn, device)
    print("Done!")

    # save model
    torch.save(adv_cnn.state_dict(), "models/FGSM_adv_CNN.pth")
    print("Saved PyTorch Model State to models/FGSM_adv_CNN.pth")

    test(test_dataloader, adv_cnn, loss_fn, device)
    print("Done!")
