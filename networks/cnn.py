from typing import Any

import torch.nn as nn
import torch.nn.functional as f
import torchvision.transforms as tr
from torch import optim

from utils.settings import settings


class CNN(nn.Module):
    """
    Simple convolutional classifier neural network.
    """

    def __init__(self, input_size: int, nb_classes: int):
        """
        Create a convolutional classifier neural network with 2 hidden layers (1 convolutional, 1 fully connected)

        :param input_size: The size of one item of the dataset used for the training
        :param nb_classes: Number of class to classify
        """
        super().__init__()

        # TODO possible dynamic input size with
        #  https://discuss.pytorch.org/t/utility-function-for-calculating-the-shape-of-a-conv-output/11173/7

        # Input -> Hidden 1 (convolution)
        self.conv = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5)
        # Pooling for Hidden 1
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        # Hidden 1 (convolution) -> Hidden 2 (fc)
        self.fc1 = nn.Linear(12 * 12 * 6, settings.size_hidden_2)
        # Hidden 2 -> Output
        self.fc2 = nn.Linear(settings.size_hidden_2, nb_classes)

        self._criterion = nn.CrossEntropyLoss()
        self._optimizer = optim.SGD(self.parameters(), lr=settings.learning_rate, momentum=settings.momentum)

    def forward(self, x: Any) -> Any:
        """
        Define the forward logic.

        :param x: One input of the dataset
        :return: The output of the network
        """
        # Convolution + Max Pooling
        x = self.pool(f.relu(self.conv(x)))
        # Flatten the data for the FC layer
        x = x.view(settings.batch_size, -1)
        # Feed forward
        x = f.relu(self.fc1(x))
        # Feed forward classification
        x = self.fc2(x)
        return x

    def training_step(self, inputs: Any, labels: Any):
        """
        Define the logic for one training step.

        :param inputs: The input from the training dataset, could be a batch or an item
        :param labels: The label of the item or the batch
        :return: The loss value
        """
        # Zero the parameter gradients
        self._optimizer.zero_grad()

        # Forward + Backward + Optimize
        outputs = self(inputs)
        loss = self._criterion(outputs, labels)
        loss.backward()
        self._optimizer.step()

        return loss

    @staticmethod
    def get_transforms():
        """
        Define the data pre-processing to apply on the datasets before to use this neural network.
        """
        return tr.Compose([
            tr.ToTensor(),  # Convert to pytorch tensor
        ])
