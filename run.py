import logging
import math
import random
import sys

import numpy as np
import seaborn as sns
import torch
from torch.nn import Module
from torch.utils.data import Dataset, DataLoader

from corrupt_network import reduce_resolution

LOGGER = logging.getLogger('weights-resolution-survey')


def preparation() -> None:
    """
    Prepare the environment before all operations.
    """

    # Configure logging
    logging.basicConfig(stream=sys.stdout, format='%(asctime)s [%(levelname)s] %(message)s')
    LOGGER.setLevel(logging.INFO)

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

    # Start the training
    _train(train_dataset, test_dataset, network)

    # Start normal test
    _test(test_dataset, network)

    # Reduce the resolution of the weights
    min_value = -1
    max_value = 1
    delta = 0.5
    nb_states = (max_value - min_value) / delta
    reduce_resolution(network, min_value, max_value, delta)
    LOGGER.info(f'Network resolution decreased to {nb_states} states ({math.log2(nb_states):.2} bits)')

    # Start low resolution test
    _test(test_dataset, network)


def _train(train_dataset: Dataset, test_dataset: Dataset, network: Module) -> None:
    LOGGER.info('Start network training...')

    # Turn on the training mode of the network
    network.train()

    # Use the pyTorch data loader
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=4)
    nb_batch = len(train_loader)

    # Iterate epoch
    nb_epoch = 4
    for epoch in range(nb_epoch):
        LOGGER.info(f'Start epoch {epoch + 1:03}/{nb_epoch} ({epoch / nb_epoch * 100:05.2f}%)')

        # Iterate batches
        for i, data in enumerate(train_loader):
            LOGGER.debug(f'Start training batch {i + 1:03}/{nb_batch} ({i / nb_batch * 100:05.2f}%)')
            # Get the inputs; data is a list of [inputs, labels]
            inputs, labels = data

            # Run a training set for these data
            loss = network.training_step(inputs, labels)
            LOGGER.debug(f'Batch loss: {loss:.5f}')

    LOGGER.info('Network training competed')


def _test(test_dataset: Dataset, network: Module) -> None:
    LOGGER.info('Start network testing...')

    # Turn on the inference mode of the network
    network.eval()

    # Use the pyTorch data loader
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=True, num_workers=4)
    nb_batch = len(test_loader)

    nb_correct = 0
    nb_total = 0
    # Diable gradient for performances
    with torch.no_grad():
        # Iterate batches
        for i, data in enumerate(test_loader):
            LOGGER.debug(f'Start testing batch {i + 1:03}/{nb_batch} ({i / nb_batch * 100:05.2f}%)')
            # Get the inputs: data is a list of [inputs, labels]
            inputs, labels = data

            # Forward
            outputs = network(inputs)
            _, predicted = torch.max(outputs, 1)  # Get the index of the max value for each image of the batch

            # Count the result
            nb_total += len(labels)
            nb_correct += torch.eq(predicted, labels).sum()

    LOGGER.info(f'Test overall accuracy: {nb_correct / nb_total * 100:05.2f}%')

    LOGGER.info('Network testing competed')
