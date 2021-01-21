import numpy as np
import torch
from torch.nn import Module
from torch.utils.data import DataLoader, Dataset

from plots.misc import plot_confusion_matrix
from utils.logger import logger
from utils.output import save_results
from utils.settings import settings


def test(test_dataset: Dataset, network: Module, test_label: str = '') -> float:
    logger.info(f'Start network {test_label} testing...')

    # Turn on the inference mode of the network
    network.eval()

    # Use the pyTorch data loader
    test_loader = DataLoader(test_dataset, batch_size=settings.batch_size, shuffle=True, num_workers=4)
    nb_batch = len(test_loader)
    nb_classes = len(test_dataset.classes)

    nb_correct = 0
    nb_total = 0
    nb_labels_predictions = np.zeros((nb_classes, nb_classes))
    # Diable gradient for performances
    with torch.no_grad():
        # Iterate batches
        for i, data in enumerate(test_loader):
            # Get the inputs: data is a list of [inputs, labels]
            inputs, labels = data

            # Forward
            outputs = network(inputs)
            _, predicted = torch.max(outputs, 1)  # Get the index of the max value for each image of the batch

            # Count the result
            nb_total += len(labels)
            nb_correct += torch.eq(predicted, labels).sum()
            for label, pred in zip(labels, predicted):
                nb_labels_predictions[label][pred] += 1
            # logger.debug(f'Start testing batch {i + 1:03}/{nb_batch} ({(i + 1) / nb_batch * 100:05.2f}%)')

    accuracy = float(nb_correct / nb_total)
    classes_accuracy = {test_dataset.classes[i]: l[i] / np.sum(l) for i, l in enumerate(nb_labels_predictions)}
    logger.info(f'Test overall accuracy {test_label}: {accuracy * 100:05.2f}%')
    logger.info(f'Test accuracy {test_label} per classes:\n\t' +
                "\n\t".join([f'{cls}: {accuracy * 100:05.2f}%' for cls, accuracy in classes_accuracy.items()]))

    logger.info(f'Network {test_label} testing competed')

    results = {f'accuracy_{test_label}': accuracy, f'classes_accuracy_{test_label}': classes_accuracy}
    save_results(**results)
    plot_confusion_matrix(nb_labels_predictions, class_names=test_dataset.classes, title_add=test_label,
                          annotations=False)

    return accuracy
