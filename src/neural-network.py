# %%
# Dataset: stores the samples and their corresponding labels
# DataLoader: wraps an iterable around the Dataset
import torch
from matplotlib import pyplot as plt
from sklearn.datasets import make_moons
from torch import nn
from torch.optim import SGD
from torch.utils.data import DataLoader, TensorDataset, random_split

# %% Generate a dataset
X, y = make_moons(n_samples=1000, noise=0.2)

# X ist a 2d vector of the x, y coordinates of the samples
# y is a 1d vector of the class labels (0 or 1) for each sample
print("X:", X.shape)
print("sample: ", X[0])
print()

print("y:", y.shape)
print("sample: ", y[0])

# %% Visualize the dataset
plt.scatter(X[:, 0], X[:, 1], c=y)
plt.show()

# %% Create tensors from the vectors
X_tensor = torch.tensor(X, dtype=torch.float32)
y_tensor = torch.tensor(y, dtype=torch.long)

# X_tensor has size [1000, 2] which means 2 dimensionsal tensor with 1000 samples and 2 features (x, y coordinates)
print("X_tensor:", X_tensor.shape, X_tensor.dtype, X_tensor.device)

# X_tensor[0] has size [2] which means 1 dimensional tensor with 2 elements
print("sample: ", X_tensor[0], ",", X_tensor[0].shape)

# X_tensor[0][0] has size [] which means a scalar value (0-dimensional tensor)
print("\t", X_tensor[0][0], ", ", X_tensor[0][0].shape)
print()

print("y_tensor:", y_tensor.shape, y_tensor.dtype, y_tensor.device)
print("sample: ", y_tensor[0])

# %% Create a TensorDataset from the vectors
# TensorDataset wraps tensors together like a map / zip function
# -> map X and y together so that when we iterate through the dataset,
#    we get both the features and the labels for each sample
dataset = TensorDataset(X_tensor, y_tensor)

# create a train test split of the dataset
train_set, test_set = random_split(dataset, [0.8, 0.2])
print(train_set[0])

# %%
# Create a DataLoader from the dataset
# dataset: abstract class, basically anything that maps indices to data samples
# batch_size: how many samples per batch to load
# shuffle: whether to shuffle the data at every epoch
train_loader = DataLoader(dataset=train_set, batch_size=32, shuffle=True)

# DataLoader is an iterable, so we can loop through it to get batches of data
# or access the iterator directly to get the next batch of data
batch = train_loader._get_iterator()._next_data()

# we set the batch size to 32, so we get 32 samples for each _next_data call
# the batch has the shape [1, 1]:
#   -> the first element is the tensor of all features: [32, 2]
#   -> the second element is the tensor of all labels:  [32]
print(batch[0].shape)
print(batch[1].shape)

# when shuffle=False, the first batch contains exactly the first 32 samples of the dataset
# print(batch[0][0] == dataset[0][0])

# when shuffle=True, each batch contains random samples from the dataset
# it guarantees that all samples are seen once per epoch, but the order is different each time
print(batch[0][0] == train_set[0][0])

# %% use hardware acceleration if available
device = (
    torch.accelerator.current_accelerator().type
    if torch.accelerator.is_available()
    else "cpu"
)
print(f"Using {device} device")


# %% Create a neural network
# nn: provides all the building blocks for creating *neural networks*
# Module: the base class for all neural network modules
class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        # Sequential: a container that chains layers together
        #   -> each layer's output is the next layer's input
        #   -> no need to write a forward method manually
        self.stack = nn.Sequential(
            nn.Linear(2, 16),  # input layer: 2 features -> 16 neurons
            nn.ReLU(),  # activation: introduces non-linearity
            nn.Linear(16, 2),  # output layer: 16 neurons -> 2 classes
        )

    # we implement the forward method to define the forward pass
    def forward(self, x):
        return self.stack(x)


# move the model on the torch device (GPU or CPU)
# this is necessary because the model and the data need to be on the same device for the computations to work
model = NeuralNetwork().to(device)
print(model)

# %% Define loss function and optimizer
# CrossEntropyLoss: standard loss for multi-class classification, combines LogSoftmax and NLLLoss
# actually: CrossEntropyLoss = -log(p_c), where p_c is the predicted probability of the correct class
# the LogSoftmax part is the log(softmax(x)) which converts the raw logits into log probabilities
# the NLLLoss part just negates the log probability, which makes the loss positive and easy to optimize/minimize
loss_fn = nn.CrossEntropyLoss()

# SGD: stochastic gradient descent,
# -> it updates the model's parameters in the direction of the negative gradient of the loss with respect to the parameters
# -> stochastic: it uses a random subset (batch) of the data to compute the gradient at each step,
#    which makes it faster and more efficient for large datasets
# lr: learning rate (how big the steps are)
optimizer = SGD(model.parameters(), lr=0.05)

# %% Training loop
epochs = 100

# epoch: one full pass through the entire training dataset
# so the model has seen all the training samples once after one epoch
for epoch in range(epochs):
    # set model to training mode (enables dropout, batch norm etc.)
    # without this, its in evaluation mode, just doing forward passes, not updating weights
    model.train()

    # iterate over all batches of the training data = one epoch
    # as set before, each batch contains 32 samples (features and labels)
    for batch in train_loader:
        # copy the batch tensors to the same device as the model (GPU or CPU)
        # necessary because model and data need to be on the same device for the computations to work
        # the batch_X is still the same tensor, just now on the GPU (it is copied)
        batch_X, batch_y = batch[0].to(device), batch[1].to(device)
        # the address space is different for the CPU and GPU, so the data pointers are different
        # print(batch[0] is batch_X, batch[0].data_ptr(), batch_X.data_ptr()) # (False, 123, 456)

        # forward pass: compute predictions, this executes the forward method of the model
        # computes all layers in the sequential stack and returns the output logits for each class
        # shape is [32, 2] because we have 32 samples in the batch and 2 output classes (logits for each class)
        # the logit, e.g [-0.6157,  0.7036] means that the model predicts class 1 with higher probability than class 0 for that sample
        # we can pass that though a softmax to get "probabilities" or a argmax to get the predicted class (0 or 1)
        pred = model(batch_X)

        # compute the loss using the predictions and the true labels
        loss = loss_fn(pred, batch_y)

        # backward pass: the backward function computes the gradients
        # it computes all gradients for all parameters and adds them to the .grad attribute of each parameter
        # thats why it does not need to return anything
        loss.backward()

        # now we have the gradients for all parameters, we can update the weights using the optimizer
        optimizer.step()

        # we need to zero the gradients after each step, because by default,
        #   torch accumulates the gradients on subsequent backward passes
        # why? because sometimes we want to accumulate gradients over multiple batches before updating the weights,
        #   for example when using larger batch sizes that do not fit in memory? or so says the AI
        optimizer.zero_grad()

    if (epoch + 1) % 10 == 0:
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss.item():.4f}")

# %% Evaluate on test set
# set model to evaluation mode (disables dropout etc.)
model.eval()

test_loader = DataLoader(dataset=test_set, batch_size=len(test_set))
with torch.no_grad():  # no gradient computation needed for evaluation
    for X_test, y_test in test_loader:
        X_test, y_test = X_test.to(device), y_test.to(device)
        pred = model(X_test)
        correct = (pred.argmax(dim=1) == y_test).sum().item()
        print(
            f"Test Accuracy: {correct}/{len(y_test)} = {correct / len(y_test) * 100:.1f}%"
        )

# %% Visualize decision boundary
with torch.no_grad():
    # create a grid of points covering the data space
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    xx, yy = torch.meshgrid(
        torch.linspace(x_min, x_max, 200),
        torch.linspace(y_min, y_max, 200),
        indexing="xy",
    )
    grid = torch.stack([xx.flatten(), yy.flatten()], dim=1).to(device)

    preds = model(grid).argmax(dim=1).cpu().reshape(xx.shape)

    plt.contourf(xx, yy, preds, alpha=0.3, cmap="coolwarm")
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap="coolwarm", edgecolors="k", s=15)
    plt.title("Decision Boundary")
    plt.show()

# %%
