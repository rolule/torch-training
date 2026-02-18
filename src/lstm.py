# %% Simple LSTM Example: Predict the next number in a sequence
# Task: Given a sequence [1, 2, 3, 4] → predict 5
#        Given a sequence [2, 3, 4, 5] → predict 6
# This is the simplest possible LSTM use case: learning a linear pattern in sequential data

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

# %% Generate training and test data
# simple sequences: [x, x+1, x+2, x+3] → target: x+4
seq_length = 4
data = torch.arange(0, 400, dtype=torch.float32)

# Normalize the data → LSTMs work much better with values around 0-1
# without normalization, large values (0-400) cause exploding/vanishing gradients
data_mean = data.mean()
data_std = data.std()

X_list = []
y_list = []
for i in range(len(data) - seq_length):
    X_list.append(data[i : i + seq_length])
    y_list.append(data[i + seq_length])

# X shape: [num_samples, seq_length, 1] → LSTM expects (batch, seq_len, input_size)
# we need the last dimension (input_size=1) because each timestep has 1 feature (the number)
X = torch.stack(X_list).unsqueeze(-1)
y = torch.stack(y_list).unsqueeze(-1)

X = (X - data_mean) / data_std
y = (y - data_mean) / data_std

# train/test split: 80/20
split = int(len(X) * 0.8)
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

print(f"Train: {X_train.shape[0]} samples, Test: {X_test.shape[0]} samples")

# %% Create DataLoaders
train_dataset = TensorDataset(X_train, y_train)
test_dataset = TensorDataset(X_test, y_test)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=len(test_dataset))


# %% Define the LSTM model
class SimpleLSTM(nn.Module):
    def __init__(self, input_size=1, hidden_size=64, output_size=1):
        super().__init__()
        # LSTM: processes sequences step by step, maintaining a hidden state
        #   input_size:  number of features per timestep (1 = just the number)
        #   hidden_size: size of the internal memory (more = more capacity)
        #   num_layers:  stacked LSTM layers (deeper = can learn more complex patterns)
        #   batch_first: input shape is (batch, seq_len, features) instead of (seq_len, batch, features)
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=2, batch_first=True)

        # Linear layer: maps the last hidden state to our prediction
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # lstm returns: (all_hidden_states, (last_hidden, last_cell))
        # out shape: [batch, seq_len, hidden_size]
        out, _ = self.lstm(x)

        # we only care about the LAST timestep's hidden state
        # out[:, -1, :] → [batch, hidden_size]
        last_hidden = out[:, -1, :]

        # map to prediction
        return self.fc(last_hidden)


model = SimpleLSTM()
print(model)

# %% Train the model
loss_fn = nn.MSELoss()  # regression task → mean squared error
optimizer = torch.optim.Adam(model.parameters(), lr=0.005)

epochs = 500
for epoch in range(epochs):
    model.train()
    epoch_loss = 0.0
    num_batches = 0
    for batch_X, batch_y in train_loader:
        pred = model(batch_X)
        loss = loss_fn(pred, batch_y)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        epoch_loss += loss.item()
        num_batches += 1

    if (epoch + 1) % 50 == 0:
        avg_loss = epoch_loss / num_batches
        # denormalize the loss to original scale for interpretability
        # MSE scales with variance (std²), so multiply by std²
        real_scale_loss = avg_loss * (data_std.item() ** 2)
        print(
            f"Epoch {epoch + 1}/{epochs}, "
            f"Loss (normalized): {avg_loss:.6e}, "
            f"Loss (original scale): {real_scale_loss:.2f}"
        )

# %% Evaluate on test set
model.eval()

with torch.no_grad():
    for X_batch, y_batch in test_loader:
        preds_norm = model(X_batch)

        # denormalize back to original scale
        preds = preds_norm * data_std + data_mean
        targets = y_batch * data_std + data_mean

        mae = (preds - targets).abs().mean()
        mse = nn.functional.mse_loss(preds, targets)

        print("Sample predictions (predicted → expected):")
        for i in range(min(10, len(preds))):
            print(f"  {preds[i].item():7.2f} → {targets[i].item():7.2f}")

        print(f"\nTest MAE:  {mae.item():.4f}")
        print(f"Test RMSE: {mse.sqrt().item():.4f}")
