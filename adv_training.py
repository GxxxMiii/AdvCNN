import random
import numpy as np
from model import *
import torch
from foolbox.models import PyTorchModel
from foolbox.v1.attacks import GradientSignAttack
from foolbox.v1.attacks import LBFGSAttack
from foolbox.v1.attacks import SaliencyMapAttack
from foolbox.v1.attacks import DeepFoolAttack
from foolbox.criteria import TargetClass
from foolbox.criteria import Misclassification


def lbfgs_adv_train(dataloader, model, loss_fn, optimizer, device):
    size = len(dataloader.dataset)
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        # turn model into foolbox model
        fmodel = PyTorchModel(model=model.eval(), bounds=(0, 255), num_classes=10)
        # adversarial training
        if batch > 100 and batch % 2 == 0:
            for i in range(len(X)):
                # L-BFGS attack
                target_class = random.randint(0, 9)
                while target_class == y[i]:
                    target_class = random.randint(0, 9)
                # print('y:', y[i].numpy(), 't:', target_class)

                criterion = TargetClass(target_class=target_class)
                attack = LBFGSAttack(model=fmodel, criterion=criterion)
                adv = attack(input_or_adv=X[i].numpy(), label=y[i])
                if adv is not None:
                    X[i] = torch.tensor(adv)

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


def fgsm_adv_train(dataloader, model, loss_fn, optimizer, device):
    size = len(dataloader.dataset)
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        # turn model into foolbox model
        fmodel = PyTorchModel(model=model.eval(), bounds=(0, 1), num_classes=10)
        # adversarial training
        if batch > 100 and batch % 2 == 0:
            for i in range(len(X)):
                # FGSM attack
                criterion = Misclassification()
                attack = GradientSignAttack(model=fmodel, criterion=criterion)
                # adv = attack(inputs=X[i].numpy()[np.newaxis, :], labels=y[i].numpy().reshape(1), epsilons=1, max_epsilon=0.2)
                adv = attack(input_or_adv=X[i].numpy(), label=y[i].numpy(), epsilons=1, max_epsilon=0.2)
                if adv is not None:
                    X[i] = torch.tensor(adv)

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


def jsma_adv_train(dataloader, model, loss_fn, optimizer, device):
    size = len(dataloader.dataset)
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        # turn model into foolbox model
        fmodel = PyTorchModel(model=model.eval(), bounds=(0, 1), num_classes=10)
        # adversarial training
        if batch > 100 and batch % 2 == 0:
            for i in range(len(X)):
                # JSMA attack
                target_class = random.randint(0, 9)
                while target_class == y[i]:
                    target_class = random.randint(0, 9)
                # print('y:', y[i].numpy(), 't:', target_class)

                criterion = TargetClass(target_class=target_class)
                attack = SaliencyMapAttack(model=fmodel, criterion=criterion)
                adv = attack(input_or_adv=X[i].numpy(), label=y[i])
                if adv is not None:
                    X[i] = torch.tensor(adv)
                    print('found adv')

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


def deepfool_adv_train(dataloader, model, loss_fn, optimizer, device):
    size = len(dataloader.dataset)
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        # turn model into foolbox model
        fmodel = PyTorchModel(model=model.eval(), bounds=(0, 1), num_classes=10)
        # adversarial training
        if batch > 100 and batch % 2 == 0:
            for i in range(len(X)):
                # Deepfool attack
                criterion = Misclassification()
                attack = DeepFoolAttack(model=fmodel, criterion=criterion)
                # adv = attack(inputs=X[i].numpy()[np.newaxis, :], labels=y[i].numpy().reshape(1), epsilons=1, max_epsilon=0.2)
                adv = attack(input_or_adv=X[i].numpy(), label=y[i].numpy())
                if adv is not None:
                    X[i] = torch.tensor(adv)
                    print('found adv')

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

    learning_rate = 0.0005
    train_dataloader, test_dataloader, device = init_data()
    adv_cnn = CNN().to(device)
    # lenet = LeNet().to(device)

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(adv_cnn.parameters(), lr=learning_rate)

    epochs = 1
    for t in range(epochs):
        print(f"Epoch {t + 1}\n-------------------------------")
        # lbfgs_adv_train(train_dataloader, adv_cnn, loss_fn, optimizer, device)
        # fgsm_adv_train(train_dataloader, adv_cnn, loss_fn, optimizer, device)
        # jsma_adv_train(train_dataloader, adv_cnn, loss_fn, optimizer, device)
        deepfool_adv_train(train_dataloader, adv_cnn, loss_fn, optimizer, device)
        test(test_dataloader, adv_cnn, loss_fn, device)
    print("Done!")

    # save model
    torch.save(adv_cnn.state_dict(), "models/LBFGS_adv_CNN.pth")
    print("Saved PyTorch Model State to models/LBFGS_adv_CNN.pth")

    test(test_dataloader, adv_cnn, loss_fn, device)
    print("Done!")
