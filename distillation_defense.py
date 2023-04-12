from model import *
import torch
import torch.nn.functional as F


def distilled_train(dataloader, model, loss_fn, optimizer, device, temp):
    """

    :param dataloader: training dataloader
    :param model: CNN model
    :param temp: distillation temperature
    """

    size = len(dataloader.dataset)
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        # Compute prediction error
        pred = model(X)
        output = F.log_softmax(pred/temp, dim=1)
        loss = loss_fn(output, y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            loss, current = loss.item(), (batch + 1) * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


def distill(train_loader, test_loader, model, d_model, epochs,
            loss_fn, teacher_optimizer, distilled_optimizer, device, temp):
    """

    :param train_loader: training dataloader
    :param test_loader: testing dataloader
    :param model: teacher CNN model
    :param d_model: distilled CNN model
    :param epochs: training epochs
    :param teacher_optimizer: optimizer for teacher model
    :param distilled_optimizer: optimizer for distilled model
    :param temp: distillation temperature
    :return: distilled model
    """

    # distilled train model in temp
    print("Training first time!")
    distilled_train(train_loader, model, loss_fn, teacher_optimizer, device, temp)
    test(test_loader, model, loss_fn, device)  # test model at temp=1
    print("Done first training!\n")
    # save intermediate model
    torch.save(model.state_dict(), "models/distillation_teacher_CNN.pth")
    print("Saved PyTorch Distillation Teacher Model State to distillation_teacher_CNN.pth")
    # model = CNN()
    # model.load_state_dict(torch.load("distillation_teacher_CNN.pth"))
    # print("Load Pytorch Model from distillation_teacher_CNN.pth")

    # convert label into soft label
    for batch, (X, y) in enumerate(train_loader):
        X, y = X.to(device), y.to(device)
        soft_label = F.log_softmax(model(X), dim=1)
        y = soft_label

    # train model again in temp
    print("Training second time!")
    for t in range(epochs):
        print(f"Epoch {t + 1}\n-------------------------------")
        distilled_train(train_loader, d_model, loss_fn, distilled_optimizer, device, temp)
        test(test_loader, d_model, loss_fn, device)  # test model at temp=1
    print("Done second training!")


if __name__ == '__main__':

    print('Defensive Distillation\n')

    learning_rate = 0.0005
    train_dataloader, test_dataloader, device = init_data()
    teacher_cnn = CNN().to(device)
    distilled_cnn = CNN().to(device)

    loss_fn = nn.CrossEntropyLoss()
    teacher_optimizer = torch.optim.Adam(teacher_cnn.parameters(), lr=learning_rate)
    distilled_optimizer = torch.optim.Adam(distilled_cnn.parameters(), lr=learning_rate)

    temp = 20

    epochs = 1
    distill(train_loader=train_dataloader, test_loader=test_dataloader, model=teacher_cnn, d_model=distilled_cnn,
            epochs=epochs, loss_fn=loss_fn, teacher_optimizer=teacher_optimizer, distilled_optimizer=distilled_optimizer,
            device=device, temp=temp)
    print("Done!")

    # save model
    torch.save(distilled_cnn.state_dict(), "models/distilled_CNN.pth")
    print("Saved PyTorch Model State to distilled_CNN.pth")

    test(test_dataloader, distilled_cnn, loss_fn, device)
    print("Done!")



