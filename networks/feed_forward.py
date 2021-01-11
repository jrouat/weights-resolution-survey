from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as f
import torchvision.transforms as tr
from torch import optim


class FeedForward(nn.Module):
    """
    Simple feed forward classifier neural network.
    """

    def __init__(self, input_size: int, nb_classes: int):
        """
        Create a new network with 2 hidden layers fully connected.

        :param input_size: The size of one item of the dataset used for the training
        :param nb_classes: Number of class to classify
        """
        super().__init__()

        self.fc1 = nn.Linear(input_size, 50)  # Input -> Hidden 1
        self.fc2 = nn.Linear(50, 50)  # Hidden 1 -> Hidden 2
        self.fc3 = nn.Linear(50, nb_classes)  # Hidden 2 -> Output

        self._criterion = nn.CrossEntropyLoss()
        self._optimizer = optim.SGD(self.parameters(), lr=0.001, momentum=0.9)

    def forward(self, x: Any) -> Any:
        """
        Define the forward logic.

        :param x: One input of the dataset
        :return: The output of the network
        """
        x = f.relu(self.fc1(x))
        x = f.relu(self.fc2(x))
        x = self.fc3(x)
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
            tr.Lambda(lambda x: torch.flatten(x))  # Flatten the 28x28 image to a 784 array
        ])
