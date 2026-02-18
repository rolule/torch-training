# Dataset: stores the samples and their corresponding labels
# DataLoader: wraps an iterable around the Dataset

import torch
from matplotlib import pyplot as plt
from sklearn.datasets import make_moons
from torch.utils.data import DataLoader, TensorDataset, random_split


def get_moon_dataset(visualize=False):
    # Generate a dataset
    X, y = make_moons(n_samples=1000, noise=0.2)

    # X ist a 2d vector of the x, y coordinates of the samples
    # y is a 1d vector of the class labels (0 or 1) for each sample
    # print("X:", X.shape)
    # print("sample: ", X[0])
    # print()

    # print("y:", y.shape)
    # print("sample: ", y[0])

    # Visualize the dataset
    if visualize:
        plt.scatter(X[:, 0], X[:, 1], c=y)
        plt.show()

    # Create tensors from the vectors
    X_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.long)

    # X_tensor has size [1000, 2] which means 2 dimensionsal tensor with 1000 samples and 2 features (x, y coordinates)
    # print("X_tensor:", X_tensor.shape, X_tensor.dtype, X_tensor.device)

    # X_tensor[0] has size [2] which means 1 dimensional tensor with 2 elements
    # print("sample: ", X_tensor[0], ",", X_tensor[0].shape)

    # X_tensor[0][0] has size [] which means a scalar value (0-dimensional tensor)
    # print("\t", X_tensor[0][0], ", ", X_tensor[0][0].shape)
    # print()

    # print("y_tensor:", y_tensor.shape, y_tensor.dtype, y_tensor.device)
    # print("sample: ", y_tensor[0])

    # Create a TensorDataset from the vectors
    # TensorDataset wraps tensors together like a map / zip function
    # -> map X and y together so that when we iterate through the dataset,
    #    we get both the features and the labels for each sample
    dataset = TensorDataset(X_tensor, y_tensor)

    # create a train test split of the dataset
    train_set, test_set = random_split(dataset, [0.8, 0.2])

    return train_set, test_set


def get_data_loaders(train_set, test_set):
    # Create DataLoaders from the datasets
    # dataset: abstract class, basically anything that maps indices to data samples
    # batch_size: how many samples per batch to load
    # shuffle: whether to shuffle the data at every epoch
    # num_workers: how many subprocesses to use for data loading, 0 means that the data will be loaded in the main process
    # -> MacOS has issues with this, so we set it to 0 for compatibility
    train_loader = DataLoader(
        dataset=train_set, batch_size=32, shuffle=True, num_workers=0
    )
    test_loader = DataLoader(dataset=test_set, batch_size=32, num_workers=0)

    # DataLoader is an iterable, so we can loop through it to get batches of data
    # or access the iterator directly to get the next batch of data
    # batch = train_loader._get_iterator()._next_data()

    # we set the batch size to 32, so we get 32 samples for each _next_data call
    # the batch has the shape [1, 1]:
    #   -> the first element is the tensor of all features: [32, 2]
    #   -> the second element is the tensor of all labels:  [32]
    # print(batch[0].shape)
    # print(batch[1].shape)

    # when shuffle=False, the first batch contains exactly the first 32 samples of the dataset
    # print(batch[0][0] == dataset[0][0])

    # when shuffle=True, each batch contains random samples from the dataset
    # it guarantees that all samples are seen once per epoch, but the order is different each time
    # print(batch[0][0] == train_set[0][0])

    return train_loader, test_loader


def get_device():
    device = (
        torch.accelerator.current_accelerator().type
        if torch.accelerator.is_available()
        else "cpu"
    )
    print(f"Using {device} device")
    return device


def visualize_decision_boundary(model, data_set, device):
    # extract raw tensors from the Subset (random_split returns Subsets)
    # .dataset gives the underlying TensorDataset, .indices the selected indices
    X = data_set.dataset.tensors[0][data_set.indices].to(device)
    y = data_set.dataset.tensors[1][data_set.indices].to(device)
    with torch.no_grad():
        # create a grid of points covering the data space
        x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
        y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
        xx, yy = torch.meshgrid(
            torch.linspace(x_min.item(), x_max.item(), 200),
            torch.linspace(y_min.item(), y_max.item(), 200),
            indexing="xy",
        )
        grid = torch.stack([xx.flatten(), yy.flatten()], dim=1).to(device)

        preds = model(grid).argmax(dim=1).cpu().reshape(xx.shape)

        plt.contourf(xx, yy, preds, alpha=0.3, cmap="coolwarm")
        plt.scatter(
            X[:, 0].cpu(),
            X[:, 1].cpu(),
            c=y.cpu(),
            cmap="coolwarm",
            edgecolors="k",
            s=15,
        )
        plt.title("Decision Boundary")
        plt.show()
