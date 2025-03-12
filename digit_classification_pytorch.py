# Requirements for webapp

pip install gradio -q

import numpy as np
import gradio as gr
from PIL import Image, ImageDraw

# Importing necessary libraries

from torchvision import datasets
from torchvision import transforms

from torch.utils.data import DataLoader

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

# Mounting google drive to safely store the minst data for futture use

from google.colab import drive
drive.mount('/content/drive')

mnist_path = '/content/drive/My Drive/mnist'

# Loading train and test data and storing it into the drive (if download is True) or loading the stored date form the drive (if download is False)

transform = transforms.Compose([
    transforms.ToTensor(),    # Convert images to tensor
    transforms.Normalize((0.5,), (0.5,))    # Normalize: mean=0.5, std=0.5
])

train_data = datasets.MNIST(
    root = mnist_path,
    train = True,
    transform = transform,
    download = False
)

test_data = datasets.MNIST(
    root = mnist_path,
    train = False,
    transform = transform,
    download = False
)

# Defining the DataLoader objects which will be used to later load the data in batches in the CNN model

loaders = {
    "train" : DataLoader(
        train_data,
        batch_size = 128,
        shuffle = True,
        num_workers = 1
    ),
    "test" : DataLoader(
        test_data,
        batch_size = 128,
        shuffle = True,
        num_workers = 1
    )
}

loaders["train"]

# Defining the CNN

class CNN(nn.Module):

  def __init__(self):
    super(CNN, self).__init__()

    self.conv1 = nn.Conv2d(1, 6, kernel_size=5, padding=2)
    self.conv2 = nn.Conv2d(6, 16, kernel_size=5)
    self.fc1 = nn.Linear(16*5*5, 120)
    self.fc2 = nn.Linear(120, 84)
    self.fc3 = nn.Linear(84, 10)

  def forward(self, X):
    X = F.max_pool2d(F.relu(self.conv1(X)), kernel_size=2, stride=2)    # Applying the first convolutions layer with relu activation and max pooling
    X = F.max_pool2d(F.relu(self.conv2(X)), kernel_size=2, stride=2)    # Applying the second convolutions layer with relu activation and max pooling
    X = X.view(-1, 16*5*5)    # Flattening
    X = F.relu(self.fc1(X))   # Applying the first dense layer with relu activation
    X = F.relu(self.fc2(X))   # Applying the second dense layer with relu activation
    X = self.fc3(X)   # Applying the third dense layer

    return F.softmax(X)

# Checking if gpu is available or not

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

device

model = CNN().to(device)

# Defining the optimizer and loss function

optimizer = optim.Adam(model.parameters(), lr = 0.001)

loss_fn = nn.CrossEntropyLoss()

# print({name: param.grad for name, param in model.named_parameters() if param.grad is not None})

def train(epoch):
  model.train()

  for batch_idx, (data, target) in enumerate(loaders["train"]):
    data, target = data.to(device), target.to(device)
    optimizer.zero_grad()

    output = model(data)    # Forward pass (getting model output)

    loss = loss_fn(output, target)    # Loss calculation

    loss.backward()    # Back propogation (calculating gradients which are then stored in .grad of model parameter)

    optimizer.step()    # Weight updation (updating the weights using the gradients in .grad)

    if batch_idx % 25 == 0:
            print(f"Train Epoch: {epoch} [{batch_idx * len(data)}/{len(loaders['train'].dataset)} ({100. * batch_idx / len(loaders['train']):.0f}%)]\tLoss: {loss.item():.6f}")

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
    print(f"\nTest set: Average loss: {test_loss:.4f}, Accuracy: {correct}/{len(loaders['test'].dataset)} ({100. * correct / len(loaders['test'].dataset):.0f}%)\n")

epoch = 10

for i in range(1, epoch+1):
  train(i)
  test()

# torch.save(model.state_dict(), '/content/drive/My Drive/lenet_cnn.pth')    # Saving the model

# # Loading the saved model

# model.load_state_dict(torch.load('/content/drive/My Drive/lenet_cnn.pth', map_location=torch.device(device)))
# model.to(device)

# model.eval()    # Setting the model to prediction model

# Definig the pararmenters to of input, dummy input to simulate the shape of real inputs (batch_size=1, channels=1, 28x28 image)

dummy_input = torch.randn(1, 1, 28, 28)
dummy_input = dummy_input.to(device)

# Converting the model(python object) to a Torchscript version

traced_model = torch.jit.trace(model, dummy_input)

# Saving

traced_model.save("/content/drive/My Drive/lenet_cnn.pth")

model = torch.jit.load("/content/drive/My Drive/lenet_cnn.pth")
model.eval()

def predict(data):
    try:
        image = data["composite"]
        if image is None or np.sum(image) == 0:
            return "Error: No strokes detected. Please draw a digit."


        image = Image.fromarray(image[:, :, 3])   # Convert to grayscale using the alpha channel

        image = image.resize((28, 28)).convert("L")   # Resize to 28x28 and convert to grayscale (1 channel)

        # Normalize and convert to PyTorch tensor

        image = np.array(image, dtype=np.float32) / 255.0
        image = torch.tensor(image, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        image = image.to(device)

        # Get model predictions

        with torch.no_grad():
            output = model(image)
            probabilities = F.softmax(output, dim=1)

            probabilities = probabilities.squeeze(0).tolist()

            # Create a dictionary mapping each digit (as a string) to its probability percentage
            result = {str(i): prob for i, prob in enumerate(probabilities)}

        return result

    except Exception as e:
        return f"Error: {str(e)}"


# Creating Gradio Interface

interface = gr.Interface(
    fn=predict,
    inputs=gr.Sketchpad(width=560, height=560, brush=gr.Brush(default_size=25)),
    outputs=gr.Label(num_top_classes=3),
    title="LeNet Handwritten Digit Classifier",
    description="Draw a digit and press 'Submit' to classify it.",
    theme="dark"
)

interface.launch(share=True)