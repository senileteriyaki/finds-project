import torch
from torchvision.datasets import MNIST
from torchvision.transforms import v2
import numpy as np
import classifier1
import classifier2
from matplotlib import pyplot as plt

N = 1000 #calibration
M = 2000 #test
n = 3000
alpha = 0.05

model = classifier1.Net()
model.load_state_dict(torch.load("models/mnist_bad.pth"))
model.eval()

data = MNIST(
    root="data",
    download=True,
    train=False,
    transform=v2.ToTensor()
)



images = torch.stack([data[i][0] for i in range(n)])
labels = data.targets.numpy()[:n]

with torch.no_grad():
    logits = model(images)
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

print(empirical_coverage) #0.9465 


worst = np.argmin(test_scores[np.arange(M), test_labels])
print(worst)

fig, (ax_img, ax_bar) = plt.subplots(1, 2, figsize=(10, 5))

ax_img.imshow(test_images[worst].reshape((28, 28)), cmap="gray")
ax_img.axis("off")

prediction_set_numbers = np.where(prediction_sets[worst])[0]
print(prediction_set_numbers)
print(test_labels[worst])
print(test_scores[worst])

ax_img.set_title(f"Coverage: {empirical_coverage:.4f}\nPrediction set: {prediction_set_numbers.tolist()}")

set_sizes = prediction_sets.sum(axis=1)
sizes, counts = np.unique(set_sizes, return_counts=True)
print(sizes, counts)

ax_bar.bar(sizes, counts, width=0.5, color='crimson', ec='black')

plt.savefig('graphs/conformal_simple_badmodel.png')
plt.show()