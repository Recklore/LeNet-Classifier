import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from src.model import CNN

# If using Google Colab, uncomment these lines:
# from google.colab import drive
# drive.mount('/content/drive')

# Define path for MNIST data (adjust as needed)
mnist_path = '/content/drive/My Drive/mnist'  # or use a local path like './data'

# Define transformations and load datasets
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

train_data = datasets.MNIST(
    root=mnist_path,
    train=True,
    transform=transform,
    download=False  # Set to True if data is not available locally
)

test_data = datasets.MNIST(
    root=mnist_path,
    train=False,
    transform=transform,
    download=False
)

# DataLoader objects for batching
loaders = {
    "train": DataLoader(train_data, batch_size=128, shuffle=True, num_workers=1),
    "test": DataLoader(test_data, batch_size=128, shuffle=True, num_workers=1)
}

# Setup device and initialize the model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CNN().to(device)

# Optimizer and loss function
optimizer = optim.Adam(model.parameters(), lr=0.001)
loss_fn = nn.CrossEntropyLoss()

def train(epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(loaders["train"]):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = loss_fn(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % 25 == 0:
            print(f"Train Epoch: {epoch} [{batch_idx * len(data)}/{len(loaders['train'].dataset)} "
                  f"({100. * batch_idx / len(loaders['train']):.0f}%)]\tLoss: {loss.item():.6f}")

def test():
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in loaders["test"]:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += loss_fn(output, target).item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
    test_loss /= len(loaders["test"].dataset)
    print(f"\nTest set: Average loss: {test_loss:.4f}, Accuracy: {correct}/{len(loaders['test'].dataset)} "
          f"({100. * correct / len(loaders['test'].dataset):.0f}%)\n")

if __name__ == '__main__':
    epochs = 10
    for epoch in range(1, epochs + 1):
        train(epoch)
        test()

    # Convert model to TorchScript and save it
    dummy_input = torch.randn(1, 1, 28, 28).to(device)
    traced_model = torch.jit.trace(model, dummy_input)

    # Ensure the models/ directory exists
    os.makedirs("models", exist_ok=True)
    model_save_path = os.path.join("models", "lenet_cnn.pth")
    traced_model.save(model_save_path)
    print(f"Model saved to {model_save_path}")