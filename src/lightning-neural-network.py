# %%
import pytorch_lightning as pl
from torch import nn
from torch.optim import SGD

from utils import (
    get_data_loaders,
    get_device,
    get_moon_dataset,
    visualize_decision_boundary,
)

device = get_device()
train_set, test_set = get_moon_dataset()
train_loader, test_loader = get_data_loaders(train_set, test_set)


# %% Create a neural network using torch lightning
# lightning: a high-level framework for PyTorch that abstracts away the training loop and other boilerplate code,
#   making it easier to write clean and efficient code for training neural networks
class LightningNeuralNetwork(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(2, 16),
            nn.ReLU(),
            nn.Linear(16, 2),
        )
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.loss_fn(y_hat, y)
        return loss

    def configure_optimizers(self):
        return SGD(self.parameters(), lr=0.05)


model = LightningNeuralNetwork().to(device)
print(model)

# %% Train the model
# for training, we need to create a Trainer object from lightning, which handles the training loop and other details for us
trainer = pl.Trainer(max_epochs=100)
trainer.fit(model, train_loader)

# %% Visualize decision boundary
# after trainer.fit(), the model may be on a different device than expected
# so we explicitly move it to our target device
model = model.to(device)
visualize_decision_boundary(model, train_set, device)
