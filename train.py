from torch.nn import Module
from torch.utils.data import Dataset, DataLoader

from plots.misc import plot_losses
from plots.parameters import parameters_distribution
from utils.logger import logger
from utils.output import load_network, save_network
from utils.settings import settings


def train(train_dataset: Dataset, test_dataset: Dataset, network: Module) -> None:
    # If path set, try to load a pre trained network from cache
    if settings.trained_network_cache_path and load_network(network, settings.trained_network_cache_path):
        return  # Stop here if the network parameters are successfully loaded from cache file

    logger.info('Start network training...')

    # Turn on the training mode of the network
    network.train()

    # Use the pyTorch data loader
    train_loader = DataLoader(train_dataset, batch_size=settings.batch_size, shuffle=True, num_workers=4)
    nb_batch = len(train_loader)

    # Store the loss values for plot
    loss_evolution = []

    # Iterate epoch
    for epoch in range(settings.nb_epoch):
        logger.info(f'Start epoch {epoch + 1:03}/{settings.nb_epoch} ({epoch / settings.nb_epoch * 100:05.2f}%)')

        # Iterate batches
        for i, data in enumerate(train_loader):

            # Get the inputs; data is a list of [inputs, labels]
            inputs, labels = data

            # Run a training set for these data
            loss = network.training_step(inputs, labels)
            loss_evolution.append(float(loss))
            # logger.debug(f'Training batch {i + 1:03}/{nb_batch} ({(i + 1) / nb_batch * 100:05.2f}%) - Loss: {loss:.5f}')

    # Post train plots
    plot_losses(loss_evolution)
    parameters_distribution(network, 'after training')

    logger.info('Network training competed')

    if settings.save_network:
        save_network(network, 'trained_network')
