# %%
import torch
from torch import nn
from torch.optim import SGD

from utils import (
    get_data_loaders,
    get_device,
    get_moon_dataset,
    visualize_decision_boundary,
)

train_set, test_set = get_moon_dataset()
train_loader, test_loader = get_data_loaders(train_set, test_set)
device = get_device()


# %% Create a neural network
# nn: provides all the building blocks for creating neural networks
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

    # implement the forward method to define the forward pass
    # we just pass the input through the sequential stack of layers and return the output
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
with torch.no_grad():  # no gradient computation needed for evaluation
    for X_test, y_test in test_loader:
        X_test, y_test = X_test.to(device), y_test.to(device)
        pred = model(X_test)
        correct = (pred.argmax(dim=1) == y_test).sum().item()
        print(
            f"Test Accuracy: {correct}/{len(y_test)} = {correct / len(y_test) * 100:.1f}%"
        )

# %% Visualize decision boundary
visualize_decision_boundary(model, train_set, device)
