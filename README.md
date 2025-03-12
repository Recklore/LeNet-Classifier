# LeNet CNN for Handwritten Digit Classification

This repository contains the implementation of a **Convolutional Neural Network (CNN) based on the LeNet architecture** for classifying handwritten digits from the MNIST dataset. The project includes data loading, training, model saving, and a web application for real-time digit classification using **Gradio**.

## 📌 Features

- Implementation of the **LeNet CNN** architecture using **PyTorch**
- **ReLU** activation instead of **Sigmoid** (a modification to the original LeNet architecture)
- Training on the **MNIST** dataset
- Saving and loading trained models
- Deployment of a **Gradio** web app for user interaction

## 📂 Project Structure

```
├── models
│   ├── lenet_cnn.pth              # Saved TorchScript model
│
├── src
│   ├── model.py                   # CNN model definition
│   ├── train.py                   # Training and evaluation functions
│   ├── predict.py                 # Model inference logic
│   ├── app.py                     # Gradio web app implementation
│   ├── __init__.py                # Package initialization
│
├── requirements.txt               # Required dependencies
├── README.md                      # Project documentation
```

## 🛠 Installation

### 1️⃣ Clone the repository

```bash
git clone https://github.com/your-username/your-repo.git
cd your-repo
```

### 2️⃣ Install dependencies

```bash
pip install -r requirements.txt
```

## 🚀 Training the Model

To train the model, run:

```bash
python src/train.py
```

This will train the model on the MNIST dataset and save the trained model in the `models/` folder.

## 🎯 Running the Web App

To launch the Gradio web application, run:

```bash
python src/app.py
```

This will start a local web server where you can draw digits and see predictions.

## 📊 Model Performance

After training for **10 epochs**, the model achieves high accuracy on the MNIST test dataset.

## 🔗 Links

- **Colab Notebook**: https://drive.google.com/file/d/1jAadjQCZm5wMC4ucFncX0M_KLO_PryMw/view?usp=sharing
- **Hugging Face Spaces**: https://huggingface.co/spaces/aman-rathour/lenet-classifier


Feel free to raise an issue if u have any suggestions
