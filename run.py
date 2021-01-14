import math
import random

import numpy as np
import seaborn as sns
import torch
from torch.nn import Module
from torch.utils.data import Dataset

from corrupt_network import reduce_resolution
from plots.parameters import parameters_distribution
from test import test
from train import train
from utils.logger import logger
from utils.results_output import init_out_directory, save_results
from utils.settings import settings


def preparation() -> None:
    """
    Prepare the environment before all operations.
    """

    # Settings are automatically loaded with the first import

    # Load logger
    logger.setLevel(settings.logger_output_level)

    # Set random seeds for reproducibility
    random.seed(42)
    torch.manual_seed(42)
    np.random.seed(42)

    # Set plot style
    sns.set_theme()

    # Print settings
    logger.info(settings)

    # Create the output directory to save results and plots
    init_out_directory()


def run(train_dataset: Dataset, test_dataset: Dataset, network: Module, device=None) -> None:
    """
    Run the training and the testing og the network.

    :param train_dataset: The training dataset
    :param test_dataset: The testing dataset
    :param network: The neural network to train
    :param device: The device to use for pytorch (None = auto)
    """
    # Automatically chooses between CPU and GPU if not specified
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Send the network to the selected device (CPU or CUDA)
    network.to(device)

    # Plots pre train
    parameters_distribution(network, 'before training')

    # Start the training
    train(train_dataset, test_dataset, network)

    # Start normal test
    accuracy = test(test_dataset, network)
    save_results(accuracy=accuracy)

    # Reduce the resolution of the weights
    nb_states = (settings.max_value - settings.min_value) / settings.inaccuracy_value
    reduce_resolution(network, settings.min_value, settings.max_value, settings.inaccuracy_value)
    logger.info(f'Network resolution decreased to {nb_states:.2} states ({math.log2(nb_states):.2} bits)')

    # Plots post resolution reduction
    parameters_distribution(network, 'after resolution reduction')

    # Start low resolution test
    accuracy_low_res = test(test_dataset, network)
    save_results(accuracy_low_res=accuracy_low_res)
