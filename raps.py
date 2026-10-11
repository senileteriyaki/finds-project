import torch
from torchvision.datasets import MNIST
from torchvision.transforms import v2
import numpy as np
import classifier1
import classifier2
from matplotlib import pyplot as plt


def ncscore(scores, labels, k, lam):
    reg = np.array(k*[0,] + (scores.shape[1] - k)*[lam,])[np.newaxis,:]  # regularization vector
    order = np.flip(scores.argsort(axis=1), axis=1)
    sortedscores = np.sort(scores, axis=1, descending=True)
    regscores = sortedscores + reg
    labelpos = np.where(order == labels[:, np.newaxis])[1]  # position of label
    return regscores.cumsum(axis=1)[np.arange(len(labels)), labelpos] - np.random.rand(len(labels)) * sortedscores[np.arange(len(labels)), labelpos]  # random number between cumulative sum up to true label - 1 and label


def run(alpha, N, labels, idx, scores, k_reg, lambda_reg):
    calibration_labels = labels[idx]
    calibration_scores = scores[idx]
    test_scores = scores[~idx]
    test_labels = labels[~idx]
    M = test_scores.shape[0]
    classes = scores.shape[1]

    ncscores = ncscore(calibration_scores, calibration_labels, k_reg, lambda_reg)
    quantile = np.ceil((N + 1) * (1 - alpha)) / N
    q_hat = np.quantile(ncscores, quantile, method="higher")

    reg = np.array(k_reg*[0,] + (classes - k_reg)*[lambda_reg,])[np.newaxis,:]
    test_sortedscores = np.sort(test_scores, axis=1, descending=True)

    test_order = np.flip(test_scores.argsort(axis=1), axis=1)
    test_regscores = test_sortedscores + reg
    test_partialsums = np.cumsum(test_regscores, axis=1)
    sorted_sets = (test_partialsums - np.random.rand(M, 1)*test_regscores) <= q_hat
    prediction_sets = np.take_along_axis(sorted_sets, test_order.argsort(axis=1), axis=1)  # revert the pi permutation

    empirical_coverage = prediction_sets[np.arange(prediction_sets.shape[0]), test_labels].mean()
    set_sizes = prediction_sets.sum(axis=1)
    meansize = set_sizes.mean()

    return (empirical_coverage, meansize)


def makegraphs(idx, bad_scores, good_scores, labels, N, k_reg, lambda_reg):
    alphas = np.arange(0.05, 0.30, 0.05)
    x = 1 - alphas

    bad_results = np.array([
        run(alpha, N, labels, idx, bad_scores, k_reg, lambda_reg) for alpha in alphas
    ])
    good_results = np.array([
        run(alpha, N, labels, idx, good_scores, k_reg, lambda_reg) for alpha in alphas
    ])

    # run() returns (coverage, set_size)
    bad_coverage, bad_set_size = bad_results[:, 0], bad_results[:, 1]
    good_coverage, good_set_size = good_results[:, 0], good_results[:, 1]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Set size vs. 1 - alpha
    axes[0].plot(x, bad_set_size, marker='o', linestyle='-',
                 label='Lower Accuracy Model')
    axes[0].plot(x, good_set_size, marker='s', linestyle='-',
                 label='Higher Accuracy Model')
    axes[0].set_xlabel(r'$1 - \alpha$')
    axes[0].set_ylabel('Set Size')
    axes[0].set_title(r'Set Size vs. $1 - \alpha$')
    axes[0].legend()
    axes[0].grid(True, linestyle='--', alpha=0.5)

    # Coverage vs. 1 - alpha
    axes[1].plot(x, bad_coverage, marker='o', linestyle='-',
                 label='Lower Accuracy Model')
    axes[1].plot(x, good_coverage, marker='s', linestyle='-',
                 label='Higher Accuracy Model')
    axes[1].set_xlabel(r'$1 - \alpha$')
    axes[1].set_ylabel('Coverage')
    axes[1].set_title(r'Coverage vs. $1 - \alpha$')
    axes[1].legend()
    axes[1].grid(True, linestyle='--', alpha=0.5)

    plt.tight_layout()
    plt.savefig('graphs/conformal_regularized_comparison.png')
    plt.show()


N = 1000  # calibration
M = 2000  # test
n = 3000
k_reg = 3
lambda_reg = 0.1

badmodel = classifier1.Net()
badmodel.load_state_dict(torch.load("models/mnist_bad.pth"))
badmodel.eval()

goodmodel = classifier2.Net()
goodmodel.load_state_dict(torch.load("models/mnist_good.pth"))
goodmodel.eval()

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
    bad_scores = torch.softmax(logits, dim=1).numpy()
    logits = goodmodel(images)
    good_scores = torch.softmax(logits, dim=1).numpy()

idx = np.array([1] * N + [0] * M) > 0
np.random.shuffle(idx)
makegraphs(idx, bad_scores, good_scores, labels, N, k_reg, lambda_reg)