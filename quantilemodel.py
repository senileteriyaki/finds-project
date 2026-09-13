from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
from torch import Tensor
from torchvision.datasets import MNIST
from torchvision.transforms import v2
from torch.utils.data import DataLoader, TensorDataset
from ucimlrepo import fetch_ucirepo

class QuantileLoss(nn.Module):
    def __init__(self, high, low):
        super(QuantileLoss, self).__init__()
        self.high = high
        self.low = low

    def loss(self, input, target, q):
        error = target - input
        return torch.maximum(q * error, (q - 1) * error).mean()

    
    def forward(self, input, target):
        return self.loss(input[:,1], target, self.high) + self.loss(input[:,0], target, self.low)


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(122, 20),
            nn.ReLU(),
            nn.Linear(20, 2)
        )
    
    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

if __name__ == "__main__":

    communities_and_crime = fetch_ucirepo(id=183)
    rs = 42
    X = communities_and_crime.data.features.copy()
    y = communities_and_crime.data.targets.squeeze()
    X = X.drop(["state", "county", "community", "communityname", "fold"],axis=1)
    X = X.replace("?", float("nan"))
    X = X.astype(float)

    imputer = SimpleImputer(strategy="median")
    X_train, X_leftover, y_train, y_leftover = train_test_split(X, y, train_size=0.5, random_state=rs)
    X_train = imputer.fit_transform(X_train)
    X_leftover = imputer.transform(X_leftover)

    tensor_X = torch.Tensor(X_train)
    tensor_y = torch.Tensor(y_train.to_numpy())
    tensor_Xtest = torch.Tensor(X_leftover)
    tensor_ytest = torch.Tensor(y_leftover.to_numpy())
    dataset = TensorDataset(tensor_X, tensor_y)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    model = Net()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    loss_fn = QuantileLoss(0.95, 0.05)


    #Training Loop
    for epoch in range(20):
        model.train()
        for (data, crime) in dataloader:
            quantiles = model(data)
            loss = loss_fn(quantiles, crime)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


    model.eval()
    correct = 0

    with torch.no_grad():   
        quantiles = model(tensor_Xtest)

        lower = quantiles[:, 0]
        upper = quantiles[:, 1]

        correct = ((lower <= tensor_ytest) & (tensor_ytest <= upper))
        coverage = correct.float().mean().item()

    print(f"Test Coverage: {coverage * 100:.2f}%")


    print(model)

    torch.save(model.state_dict(), "models/crimebad.pth")