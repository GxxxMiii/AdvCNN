from attacks import *
from model import *
from utils import *


def cross_eva(clean_model, model, dir):
    """

    :param clean_model: un-defensed pytorch model
    :param model: pytorch model to evaluate
    :param dir: adversarial sample directory
    """

    torch.set_printoptions(precision=4, sci_mode=False)
    softmax = torch.nn.Softmax(dim=1)

    # Get cpu or gpu device for training.
    device = "cuda" if torch.cuda.is_available() else "cpu"
    clean_model.to(device)
    model.to(device)
    print(f"Using {device} device")

    samples = np.load(dir)
    advs_num = int(samples.shape[0]/2)

    i = 0
    success = 0

    for s in range(advs_num):
        image = torch.tensor(samples[i])
        adv = torch.tensor(samples[i+1])
        i += 2

        pred = clean_model(image.reshape(1, 1, 28, 28).to(device)).cpu()
        y = np.argmax(pred.detach().numpy())

        model_pred = model(adv.reshape(1, 1, 28, 28).to(device)).cpu()
        model_label = np.argmax(model_pred.detach().numpy())

        if model_label == y:
            success += 1

    print()
    print(f"adversarials num: {advs_num}")
    print(f"defense ratio: {success/advs_num:.4f}")


if __name__ == '__main__':

    # load model
    clean_cnn = CNN().eval()
    clean_cnn.load_state_dict(torch.load("models/clean_CNN.pth"))
    print("Load Pytorch Model from models/clean_CNN.pth\n")

    cnn = CNN().eval()
    cnn.load_state_dict(torch.load("models/regularized_L1000_CNN.pth"))
    print("Load Pytorch Model from models/regularized_L1000_CNN.pth\n")

    dir = 'pics/'
    print("adversarial examples directory:", dir[:-1])

    print('\nLBFGS')
    cross_eva(clean_model=clean_cnn, model=cnn, dir=dir+'lbfgs.npy')
    print('\nFGSM')
    cross_eva(clean_model=clean_cnn, model=cnn, dir=dir+'fgsm.npy')
    print('\nJSMA')
    cross_eva(clean_model=clean_cnn, model=cnn, dir=dir+'jsma.npy')
    print('\nDeepfool')
    cross_eva(clean_model=clean_cnn, model=cnn, dir=dir+'deepfool.npy')

