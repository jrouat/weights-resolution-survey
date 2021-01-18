from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as f
import torchvision.transforms as tr
from torch import optim, Tensor

from utils.settings import settings


class SNN(nn.Module):
    """
    Spiking neural network classifier with gradient descent using surrogate function.
    """

    def __init__(self, input_size: int, nb_classes: int):
        """
        Create a convolutional classifier neural network with 2 hidden layers (1 convolutional, 1 fully connected)

        :param input_size: The size of one item of the dataset used for the training
        :param nb_classes: Number of class to classify
        """
        super().__init__()

        self.fc1 = nn.Linear(input_size, settings.size_hidden_1)  # Input -> Hidden 1
        self.fc2 = nn.Linear(settings.size_hidden_1, settings.size_hidden_2)  # Hidden 1 -> Hidden 2
        self.fc3 = nn.Linear(settings.size_hidden_2, nb_classes)  # Hidden 2 -> Output

        self._criterion = nn.MSELoss(reduction='mean')
        self._optimizer = optim.Adam(self.parameters(), lr=settings.learning_rate, amsgrad=True)

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
    def transform_img_to_spikes(img) -> Tensor:
        """
        Convert an image to a image to a spikes train based on the color of each pixel

        :param img: The image to convert
        """
        # We'll normalize our input data in the range [0., 1[
        img = img / pow(2, 8)  # 256
        # "Remove" the pixels associated with darker pixels (Presumably less information)
        img[img < .25] = 1
        # Conversion to the spiking time
        # The brighter the white, the earlier the spike
        img = (1 - img) * settings.absolute_duration
        # Create a value for each time step
        spikes_train = torch.zeros((len(img), settings.absolute_duration))
        for i in range(len(img)):
            # If the pixel is not white (or removed) add a spike
            if img[i] != 0:
                spikes_train[img[i]] = 1

        return spikes_train

    @staticmethod
    def get_transforms():
        """
        Define the data pre-processing to apply on the datasets before to use this neural network.
        """
        return tr.Compose([
            # Convert to pytorch tensor
            tr.ToTensor(),
            # Flatten the 28x28 image to a 784 array
            tr.Lambda(lambda x: torch.flatten(x)),
            # Transform the image to a spike train
            tr.Lambda(lambda x: SNN.transform_img_to_spikes(x))
        ])

    def run_spiking_layer(input_spike_train, layer_weights, device):
        """
        Here we implement a current-LIF dynamic in pytorch
        """

        # First, we multiply the input spike train by the weights of the current layer to get the current that will be added
        # We can calculate this beforehand because the weights are constant in the forward pass (no plasticity)
        # Equivalent to a matrix multiplication for tensors of dim > 2 using Einstein's Notation
        input_current = torch.einsum("abc,bd->adc", (input_spike_train, layer_weights))

        recorded_spikes = []  # Array of the output spikes at each time t
        membrane_potential_at_t = torch.zeros((input_spike_train.shape[0], layer_weights.shape[-1]), device=device,
                                              dtype=torch.float)
        membrane_current_at_t = torch.zeros((input_spike_train.shape[0], layer_weights.shape[-1]), device=device,
                                            dtype=torch.float)

        for t in range(p.absolute_duration):  # For every timestep
            # Apply the leak
            # Using tau_v with euler or exact method
            membrane_potential_at_t = (1 - int(p.delta_t) / int(p.tau_v)) * membrane_potential_at_t
            # Using tau_i with euler or exact method
            membrane_current_at_t = (1 - int(p.delta_t) / int(p.tau_i)) * membrane_current_at_t

            # Select the input current at time t
            input_at_t = input_current[:, :, t]

            # Integrate the input current
            membrane_current_at_t += input_at_t

            # Integrate the input to the membrane potential
            membrane_potential_at_t += membrane_current_at_t

            # Select the surrogate function based on the parameters
            spike_functions = None
            if p.surrogate_gradient == 'relu':
                spike_functions = SpikeFunctionRelu
            elif p.surrogate_gradient == 'fast_sigmoid':
                spike_functions = SpikeFunctionFastSigmoid
            elif p.surrogate_gradient == 'piecewise':
                spike_functions = SpikeFunctionPiecewise
            elif p.surrogate_gradient == 'sigmoid':
                spike_functions = SpikeFunctionSigmoid
            elif p.surrogate_gradient == 'piecewise_sym':
                spike_functions = SpikeFunctionPiecewiseSymmetric

            # Set the alpha variable
            spike_functions.alpha = p.surrogate_alpha

            # Apply the non-differentiable function
            recorded_spikes_at_t = spike_functions.apply(membrane_potential_at_t - p.v_threshold)

            recorded_spikes.append(recorded_spikes_at_t)

            # Reset the spiked neurons
            membrane_potential_at_t[membrane_potential_at_t > p.v_threshold] = 0

        recorded_spikes = torch.stack(recorded_spikes, dim=2)  # Stack over time axis (Array -> Tensor)
        return recorded_spikes
