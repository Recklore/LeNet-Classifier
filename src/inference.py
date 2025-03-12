import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image

# Setup device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_model(model_path="models/lenet_cnn.pth"):
    model = torch.jit.load(model_path, map_location=device)
    model.eval()
    return model

# Load the saved TorchScript model
model = load_model()

def predict(data):
    try:
        # Expecting 'data' to be a dictionary with key "composite"
        image = data["composite"]
        if image is None or np.sum(image) == 0:
            return "Error: No strokes detected. Please draw a digit."

        # Convert the sketch (assumed RGBA) to a grayscale image using the alpha channel
        image = Image.fromarray(image[:, :, 3])
        image = image.resize((28, 28)).convert("L")

        # Normalize and convert to a PyTorch tensor
        image = np.array(image, dtype=np.float32) / 255.0
        image = torch.tensor(image, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)

        with torch.no_grad():
            output = model(image)
            probabilities = F.softmax(output, dim=1)
            probabilities = probabilities.squeeze(0).tolist()

        # Map each digit (as a string) to its probability
        result = {str(i): prob for i, prob in enumerate(probabilities)}
        return result

    except Exception as e:
        return f"Error: {str(e)}"
