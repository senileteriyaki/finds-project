import torch
import torch.nn as nn
from torch import Tensor
from torchvision.datasets import MNIST
from torchvision.transforms import v2
from torch.utils.data import DataLoader


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28, 50),
            nn.ReLU(),
            nn.Linear(50, 10)
        )
    
    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits


traindata = MNIST(root="data", 
             download=True,
             train=True, 
             transform=v2.ToTensor())

testdata = MNIST(root="data", download=True, train=False, transform=v2.ToTensor())

model = Net()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
loss_fn = nn.CrossEntropyLoss()
dataloader = torch.utils.data.DataLoader(traindata, batch_size = 16, shuffle=True)

#Training Loop
for epoch in range(10):
    model.train()
    for (image, label) in dataloader:
        logits = model(image)
        loss = loss_fn(logits, label)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

#testing
model.eval()
correct = 0
with torch.no_grad():
    for (image, label) in testdata:
        logits = model(image)
        pred = logits.argmax(1)
        if pred.item() == label:
            correct += 1


print(model)
print(f"Test Accuracy: {correct / len(testdata) * 100:.2f}%")
torch.save(model.state_dict(), "models/mnist_good.pth")