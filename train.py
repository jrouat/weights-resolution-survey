from torch.nn import Module
from torch.utils.data import Dataset, DataLoader

from plots.misc import plot_losses
from plots.parameters import parameters_distribution
from utils.logger import logger
from utils.settings import settings


def train(train_dataset: Dataset, test_dataset: Dataset, network: Module) -> None:
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
            logger.debug(f'Start training batch {i + 1:03}/{nb_batch} ({i / nb_batch * 100:05.2f}%)')
            # Get the inputs; data is a list of [inputs, labels]
            inputs, labels = data

            # Run a training set for these data
            loss = network.training_step(inputs, labels)
            loss_evolution.append(float(loss))
            logger.debug(f'Batch loss: {loss:.5f}')

    # Post train plots
    plot_losses(loss_evolution)
    parameters_distribution(network, 'after training')

    logger.info('Network training competed')
