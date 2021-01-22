import math
import random

import numpy as np
import torch
from torch.nn import Module
from torch.utils.data import Dataset

from corrupt_network import reduce_resolution
from plots.parameters import parameters_distribution
from test import test
from train import train
from utils.logger import logger
from utils.metrics import network_metrics
from utils.output import init_out_directory, set_plot_style
from utils.settings import settings


def preparation() -> None:
    """
    Prepare the environment before all operations.
    """

    # Settings are automatically loaded with the first import

    # Setup console logger but wait to create the directory before to setup the file output
    logger.set_console_level(settings.logger_console_level)

    # Create the output directory to save results and plots
    init_out_directory()

    # Set plot style
    set_plot_style()

    # Set random seeds for reproducibility
    random.seed(42)
    torch.manual_seed(42)
    np.random.seed(42)

    # Print settings
    logger.info(settings)


def clean_up() -> None:
    """
    Clean up the environment after all operations. After that a new run can start again.
    """

    # Disable the log file, so a new one can be set later
    if settings.run_name and settings.logger_file_enable:
        logger.disable_log_file()


def run(train_dataset: Dataset, test_dataset: Dataset, network: Module, device=None) -> None:
    """
    Run the training and the testing of the network.

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

    # Save network stats and show if debug enable
    network_metrics(network, test_dataset[0][0].shape, device)

    # Plots pre train
    parameters_distribution(network, 'before training')

    # Start the training
    train(train_dataset, test_dataset, network)

    # Start normal test
    test(test_dataset, network, 'ideal')

    if settings.inaccuracy_value != 0:
        # Reduce the resolution of the weights
        reduce_resolution(network, settings.min_value, settings.max_value, settings.inaccuracy_value)
        nb_states = (settings.max_value - settings.min_value) / settings.inaccuracy_value
        logger.info(f'Network resolution decreased to {nb_states:.2} states ({math.log2(nb_states):.2} bits)')
    else:
        logger.warning('Network resolution didn\'t changed because inaccuracy value is 0')

    # Plots post resolution reduction
    parameters_distribution(network, 'after resolution reduction')

    # Start low resolution test
    test(test_dataset, network, 'low_resolution')
