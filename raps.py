import torch
from torchvision.datasets import MNIST
from torchvision.transforms import v2
import numpy as np
import classifier1
import classifier2
from matplotlib import pyplot as plt

def ncscore(scores, labels, k, lam):
    reg = np.array(k*[0,] + (scores.shape[1] - k)*[lam,])[np.newaxis,:] #regularization vector
    order = np.flip(scores.argsort(axis=1), axis=1) 
    sortedscores = np.sort(scores, axis=1, descending=True)
    regscores = sortedscores + reg 
    labelpos = np.where(order == labels[:, np.newaxis])[1] #position of label
    return regscores.cumsum(axis=1)[np.arange(len(labels)), labelpos] - np.random.rand(len(labels)) * regscores[np.arange(len(labels)), labelpos] #random number between cumulative sum up to true label - 1 and label


N = 1000 #calibration
M = 2000 #test
classes = 10
n = 3000
alpha = 0.05
k_reg = 5
lambda_reg = 0.01

model = classifier2.Net()
model.load_state_dict(torch.load("models/mnist_good.pth"))
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


ncscores = ncscore(calibration_scores, calibration_labels, k_reg, lambda_reg)

quantile = np.ceil((N + 1) * (1 - alpha)) / N
q_hat = np.quantile(ncscores, quantile, method="higher")

reg = np.array(k_reg*[0,] + (classes - k_reg)*[lambda_reg,])[np.newaxis,:]  
test_sortedscores = np.sort(test_scores, axis=1, descending=True)

test_order = np.flip(test_scores.argsort(axis=1), axis=1)
test_regscores = test_sortedscores + reg
test_partialsums = np.cumsum(test_regscores, axis=1)
sorted_sets = (test_partialsums - np.random.rand(M, 1)*test_regscores) <= q_hat
prediction_sets = np.take_along_axis(sorted_sets, test_order.argsort(axis=1), axis=1) #revert the pi permutation


empirical_coverage = prediction_sets[np.arange(prediction_sets.shape[0]), test_labels].mean()

print(empirical_coverage) 


worst = np.argmin(test_scores[np.arange(M), test_labels])
print(worst)

fig, (ax_img, ax_bar) = plt.subplots(1, 2, figsize=(10, 5))

ax_img.imshow(test_images[worst].reshape((28, 28)), cmap="gray")
ax_img.axis("off")

prediction_set_numbers = np.where(prediction_sets[worst])[0]
print(prediction_set_numbers)
print(test_labels[worst])
print(test_scores[worst])

set_sizes = prediction_sets.sum(axis=1)
sizes, counts = np.unique(set_sizes, return_counts=True)
meansize = set_sizes.mean()
print(sizes, counts)

ax_img.set_title(f"Coverage: {empirical_coverage:.4f}\nMean set size: {meansize:.2f}\n Prediction set: {prediction_set_numbers.tolist()}")




ax_bar.bar(sizes, counts, width=0.5, color='crimson', ec='black')

plt.savefig('graphs/conformal_regularized_goodmodel.png')
plt.show()
