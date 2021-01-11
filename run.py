import random

import numpy as np
import seaborn as sns
import torch
from torch.nn import Module
from torch.utils.data import Dataset, DataLoader


def preparation() -> None:
    """
    Prepare the environment before all operations.
    """

    # Set random seeds for reproducibility
    random.seed(42)
    torch.manual_seed(42)
    np.random.seed(42)

    # Set plot style
    sns.set_theme()


def run(train_dataset: Dataset, test_dataset: Dataset, network: Module, device=None) -> None:
    """
    Run the training and the testing og the network.

    :param train_dataset: The training dataset
    :param test_dataset: The testing dataset
    :param network: The neural network to train
    """
    # Automatically chooses between CPU and GPU if not specified
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Send the network to the selected device (CPU or CUDA)
    network.to(device)

    # Use the pyTorch data loader
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=2)

    # Iterate epoch
    for epoch in range(4):
        # Iterate batches
        for i, data in enumerate(train_loader):
            # Get the inputs; data is a list of [inputs, labels]
            inputs, labels = data

            # Run a training set for these data
            loss = network.training_step(inputs, labels)
