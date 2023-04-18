from attacks import *
from model import *
from utils import *


def cross_eva(model, dir):
    """

    :param model: pytorch model
    :param dir: adversarial sample directory
    """

    torch.set_printoptions(precision=4, sci_mode=False)
    softmax = torch.nn.Softmax(dim=1)

    # Get cpu or gpu device for training.
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    print(f"Using {device} device")

    samples = np.load(dir)
    advs_num = int(samples.shape[0]/2)

    i = 0
    success = 0
    confidence = 0

    for s in range(advs_num):
        image = torch.tensor(samples[i])
        adv = torch.tensor(samples[i+1])
        i += 2

        image_pred = model(image.reshape(1, 1, 28, 28).to(device)).cpu()
        image_label = np.argmax(image_pred.detach().numpy())

        adv_pred = model(adv.reshape(1, 1, 28, 28).to(device)).cpu()
        adv_label = np.argmax(adv_pred.detach().numpy())

        if image_label != adv_label:
            success += 1

            confidence += torch.max(softmax(adv_pred.detach()))

    print()
    print(f"adversarials num: {advs_num}")
    print(f"defense ratio: {(advs_num-success)/advs_num:.4f}")
    print(f"average confidence: {confidence / success:.4f}")


if __name__ == '__main__':

    # load model
    cnn = CNN().eval()
    # print(clean_cnn)
    cnn.load_state_dict(torch.load("models/distilled_CNN.pth"))
    print("Load Pytorch Model from models/distilled_CNN.pth\n")

    dir = 'pics/deepfool.npy'

    cross_eva(model=cnn, dir=dir)

