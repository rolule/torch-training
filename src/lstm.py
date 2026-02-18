# %% Simple LSTM Example: Predict the next number in a sequence
# Task: Given a sequence [1, 2, 3, 4] → predict 5
#        Given a sequence [2, 3, 4, 5] → predict 6
# This is the simplest possible LSTM use case: learning a linear pattern in sequential data

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

# %% Generate training data
# simple sequences: [x, x+1, x+2, x+3] → target: x+4
seq_length = 4
data = torch.arange(0, 20, dtype=torch.float32)

X_list = []
y_list = []
for i in range(len(data) - seq_length):
    X_list.append(data[i : i + seq_length])
    y_list.append(data[i + seq_length])

# X shape: [num_samples, seq_length, 1] → LSTM expects (batch, seq_len, input_size)
# we need the last dimension (input_size=1) because each timestep has 1 feature (the number)
X = torch.stack(X_list).unsqueeze(-1)
y = torch.stack(y_list).unsqueeze(-1)

print(f"X shape: {X.shape}")  # [16, 4, 1]
print(f"y shape: {y.shape}")  # [16, 1]
print(f"Example: {X[0].squeeze().tolist()} → {y[0].item()}")
print(f"Example: {X[5].squeeze().tolist()} → {y[5].item()}")

# %% Create DataLoader
dataset = TensorDataset(X, y)
loader = DataLoader(dataset, batch_size=4, shuffle=True)


# %% Define the LSTM model
class SimpleLSTM(nn.Module):
    def __init__(self, input_size=1, hidden_size=16, output_size=1):
        super().__init__()
        # LSTM: processes sequences step by step, maintaining a hidden state
        #   input_size:  number of features per timestep (1 = just the number)
        #   hidden_size: size of the internal memory (more = more capacity)
        #   batch_first: input shape is (batch, seq_len, features) instead of (seq_len, batch, features)
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)

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
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

epochs = 200
for epoch in range(epochs):
    model.train()
    for batch_X, batch_y in loader:
        pred = model(batch_X)
        loss = loss_fn(pred, batch_y)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    if (epoch + 1) % 20 == 0:
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss.item():.4f}")

# %% Test the model
model.eval()
with torch.no_grad():
    # test: [5, 6, 7, 8] → should predict ~9
    test_input = torch.tensor([5.0, 6.0, 7.0, 8.0]).reshape(1, 4, 1)
    pred = model(test_input)
    print(f"Input: [5, 6, 7, 8] → Predicted: {pred.item():.2f} (expected: 9)")

    # test: [10, 11, 12, 13] → should predict ~14
    test_input = torch.tensor([10.0, 11.0, 12.0, 13.0]).reshape(1, 4, 1)
    pred = model(test_input)
    print(f"Input: [10, 11, 12, 13] → Predicted: {pred.item():.2f} (expected: 14)")

    # test: [50, 51, 52, 53] → extrapolation, outside training range!
    test_input = torch.tensor([50.0, 51.0, 52.0, 53.0]).reshape(1, 4, 1)
    pred = model(test_input)
    print(f"Input: [50, 51, 52, 53] → Predicted: {pred.item():.2f} (expected: 54)")

# %%
