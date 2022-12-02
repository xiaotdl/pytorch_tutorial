"""
Optimizing Model Parameters
source: https://pytorch.org/tutorials/beginner/basics/optimization_tutorial.html
"""

# Now that we have a model and data it’s time to train, validate and test our model by optimizing its parameters on our data.
# Training a model is an iterative process; in each iteration the model 1) makes a guess about the output, 2) calculates the error in its guess (loss), 3) collects the derivatives of the error with respect to its parameters (as we saw in the previous section), and 4) optimizes these parameters using gradient descent. For a more detailed walkthrough of this process, check out this video on backpropagation from 3Blue1Brown.

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor

training_data = datasets.FashionMNIST(
    root="data",
    train=True,
    download=True,
    transform=ToTensor()
)

test_data = datasets.FashionMNIST(
    root="data",
    train=False,
    download=True,
    transform=ToTensor()
)

train_dataloader = DataLoader(training_data, batch_size=64)
test_dataloader = DataLoader(test_data, batch_size=64)

class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10),
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

model = NeuralNetwork()


print("== Hyperparameters ==")
# Hyperparameters are adjustable parameters that let you control the model optimization process. Different hyperparameter values can impact model training and convergence rates (read more about hyperparameter tuning)
# 
# We define the following hyperparameters for training:
# 1) Number of Epochs - the number times to iterate over the dataset
# 2) Batch Size - the number of data samples propagated through the network before the parameters are updated
# 3) Learning Rate - how much to update models parameters at each batch/epoch. Smaller values yield slow learning speed, while large values may result in unpredictable behavior during training.

learning_rate = 1e-3
batch_size = 64
epochs = 5

print("== Optimization Loop ==")

# Once we set our hyperparameters, we can then train and optimize our model with an optimization loop. Each iteration of the optimization loop is called an epoch.

# Each epoch consists of two main parts:
# 1) The Train Loop - iterate over the training dataset and try to converge to optimal parameters.
# 2) The Validation/Test Loop - iterate over the test dataset to check if model performance is improving.


print("== Loss Function ==")


# Common loss functions include 
# 1) nn.MSELoss (Mean Square Error) for regression tasks, and 
# 2) nn.NLLLoss (Negative Log Likelihood) for classification. 
# 3) nn.CrossEntropyLoss combines nn.LogSoftmax and nn.NLLLoss.

# Initialize the loss function
loss_fn = nn.CrossEntropyLoss()


print("== Optimizer ==")

# Optimization is the process of adjusting model parameters to reduce model error in each training step. Optimization algorithms define how this process is performed (in this example we use Stochastic Gradient Descent). All optimization logic is encapsulated in the optimizer object. Here, we use the SGD optimizer; additionally, there are many different optimizers available in PyTorch such as ADAM and RMSProp, that work better for different kinds of models and data.

optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)


print("== Full Implementation ==")

def train_loop(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    for batch, (X, y) in enumerate(dataloader):
        # Compute prediction and loss
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


def test_loop(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0

    with torch.no_grad():
        for X, y in dataloader:
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

epochs = 10
for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train_loop(train_dataloader, model, loss_fn, optimizer)
    test_loop(test_dataloader, model, loss_fn)
print("Done!")
