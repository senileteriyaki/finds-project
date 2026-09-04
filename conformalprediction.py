import torch
from torchvision.datasets import MNIST
from torchvision.transforms import v2
import numpy as np
import classifier1
import classifier2
from matplotlib import pyplot as plt

N = 1000 #calibration
M = 1000 #test
n = 2000
alpha = 0.1

badmodel = classifier1.Net()
badmodel.load_state_dict(torch.load("models/mnist_bad.pth"))
badmodel.eval()

data = MNIST(
    root="data",
    download=True,
    train=False,
    transform=v2.ToTensor()
)



images = torch.stack([data[i][0] for i in range(n)])
labels = data.targets.numpy()[:n]

with torch.no_grad():
    logits = badmodel(images)
    scores = torch.softmax(logits, dim=1).numpy()





idx = np.array([1] * N + [0] * M) > 0
np.random.shuffle(idx)

calibration_images = images[idx]
calibration_labels = labels[idx]
calibration_scores = scores[idx]
test_scores = scores[~idx]
test_images = images[~idx]
test_labels = labels[~idx]

ncscores = 1 - calibration_scores[np.arange(N), calibration_labels]
quantile = np.ceil((N + 1) * (1 - alpha)) / N
q_hat = np.quantile(ncscores, quantile, method="higher")

prediction_sets = test_scores >= (1 - q_hat)
empirical_coverage = prediction_sets[np.arange(prediction_sets.shape[0]), test_labels].mean()

print(empirical_coverage)


with torch.no_grad(): 
    logits = badmodel(calibration_images[np.argmax(ncscores)])
    print(torch.softmax(logits, dim=1))

plt.imshow(calibration_images[np.argmax(ncscores)][0], cmap="gray") #the worst image in calibration
plt.show()





