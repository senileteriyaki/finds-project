import torch
from torchvision.datasets import MNIST
from torchvision.transforms import v2
import numpy as np
import classifier2

N = 1000 #calibration
M = 1000 #test

alpha = 0.1

goodmodel = classifier2.Net()
goodmodel.load_state_dict(torch.load("models/mnist_good.pth"))
goodmodel.eval()

data = MNIST(
    root="data",
    download=True,
    train=False,
    transform=v2.ToTensor()
)

n = 2000

images = torch.stack([data[i][0] for i in range(n)])
labels = data.targets.numpy()[:n]

with torch.no_grad():
    logits = goodmodel(images)
    scores = torch.softmax(logits, dim=1).numpy()





idx = np.array([1] * N + [0] * M) > 0
np.random.shuffle(idx)

