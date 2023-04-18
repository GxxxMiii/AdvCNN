import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor


# Define model
class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28 * 28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10)
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=(5, 5), stride=(1, 1), padding=2),
            nn.BatchNorm2d(num_features=32, eps=1e-4),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(5, 5), stride=(1, 1), padding=2),
            nn.BatchNorm2d(num_features=64, eps=1e-4),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.fc = nn.Sequential(
            nn.Linear(7*7*64, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout()
        )
        self.classifier = nn.Linear(1024, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.size()[0], -1)
        x = self.fc(x)
        logits = self.classifier(x)

        return logits


def init_data():
    """

    :return: train_dataloader, test_dataloader, device
    """
    # Download training data from open datasets.
    training_data = datasets.MNIST(root="data", train=True, download=True, transform=ToTensor(),)

    # Download test data from open datasets.
    test_data = datasets.MNIST(root="data", train=False, download=True, transform=ToTensor(),)

    batch_size = 64

    # Create data loaders.
    train_dataloader = DataLoader(training_data, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=True)

    # Get cpu or gpu device for training.
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using {device} device")

    return train_dataloader, test_dataloader, device


def train(dataloader, model, loss_fn, optimizer, device):
    size = len(dataloader.dataset)
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

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


def test(dataloader, model, loss_fn, device):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.3f}%, Avg loss: {test_loss:>8f} \n")


if __name__ == '__main__':

    learning_rate = 0.0005
    train_dataloader, test_dataloader, device = init_data()
    cnn = CNN().to(device)
    # lenet = LeNet().to(device)

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(cnn.parameters(), lr=learning_rate)

    # # load model
    # clean_cnn = CNN()
    # # print(clean_cnn)
    # clean_cnn.load_state_dict(torch.load("models/clean_CNN.pth"))
    # print("Load Pytorch Model from models/clean_CNN.pth")

    epochs = 20
    for t in range(epochs):
        print(f"Epoch {t + 1}\n-------------------------------")
        train(train_dataloader, cnn, loss_fn, optimizer, device)
        test(test_dataloader, cnn, loss_fn, device)
    print("Done!")

    # save model
    torch.save(cnn.state_dict(), "models/clean_CNN.pth")
    print("Saved PyTorch Model State to models/clean_CNN.pth")

    # test(test_dataloader, clean_cnn, loss_fn, device)
    print("Done!")
