from model import *
import torch


def regularized_train(dataloader, model, loss_fn, optimizer, device, lmbda):
    """

    :param dataloader: training dataloader
    :param model: CNN model
    :param lmbda: penalty strength
    """

    size = len(dataloader.dataset)
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)
        X.requires_grad_()

        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation and gradient regularization
        optimizer.zero_grad()
        grad = torch.autograd.grad(loss, X, retain_graph=True)[0]
        grad_norm = torch.norm(grad)
        loss += lmbda * grad_norm.mean()
        loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            loss, current = loss.item(), (batch + 1) * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


if __name__ == '__main__':

    print('Gradients Regularization\n')

    learning_rate = 0.001
    train_dataloader, test_dataloader, device = init_data()
    regularized_cnn = CNN().to(device)

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(regularized_cnn.parameters(), lr=learning_rate)

    lmbda = 0.5
    epochs = 1
    for t in range(epochs):
        print(f"Epoch {t + 1}\n-------------------------------")
        regularized_train(dataloader=train_dataloader, model=regularized_cnn, loss_fn=loss_fn, optimizer=optimizer,
                          device=device, lmbda=lmbda)
        test(test_dataloader, regularized_cnn, loss_fn, device)
    print("Done!")

    # save model
    torch.save(regularized_cnn.state_dict(), "regularized_CNN.pth")
    print("Saved PyTorch Model State to regularized_CNN.pth")

    test(test_dataloader, regularized_cnn, loss_fn, device)
    print("Done!")
