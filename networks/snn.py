from typing import Any, List

import torch
import torch.nn as nn
import torchvision.transforms as tr
from torch import optim, Tensor

from utils.settings import settings


class SNN(nn.Module):
    """
    Spiking neural network classifier with gradient descent using surrogate function.
    Code adapted from https://github.com/surrogate-gradient-learning/spytorch (https://doi.org/10.1109/MSP.2019.2931595)
    """

    def __init__(self, input_size: int, nb_classes: int):
        """
        :param input_size: The size of one item of the dataset used for the training
        :param nb_classes: Number of class to classify
        """
        super().__init__()

        self.nb_classes = nb_classes

        # Create the parameters layers.
        # FIXME the input-output of each layer need to be inverted to match with the following code (especially for
        #  the 'run_spiking_layer' function). But this can probably be changed.
        self.fc1 = nn.Linear(settings.size_hidden_1, input_size, bias=False)  # Input -> Hidden 1
        self.fc2 = nn.Linear(settings.size_hidden_2, settings.size_hidden_1, bias=False)  # Hidden 1 -> Hidden 2
        self.fc3 = nn.Linear(nb_classes, settings.size_hidden_2, bias=False)  # Hidden 2 -> Output

        self._criterion = nn.MSELoss(reduction='mean')
        self._optimizer = optim.Adam(self.parameters(), lr=settings.learning_rate, amsgrad=True)

    def forward(self, input_spikes: Any) -> Any:
        """
        Define the forward logic.

        :param input_spikes: One input of the dataset
        :return: The output of the network
        """

        next_layer_input = input_spikes

        for layer_param in self.parameters():
            # Measure the spikes of layer a for each i sample
            next_layer_input = SNN.run_spiking_layer(next_layer_input, layer_param)

        # Count the spikes over time axis from the last layer output
        return torch.sum(next_layer_input, 2)

    def training_step(self, inputs: Any, labels: Any):
        """
        Define the logic for one training step.

        :param inputs: The input from the training dataset, could be a batch or an item
        :param labels: The label of the item or the batch
        :return: The loss value
        """
        # Zero the parameter gradients
        self._optimizer.zero_grad()

        # Convert labels to spikes train
        target_output = self.labels_to_target_spikes(labels)

        # Forward + Backward + Optimize
        outputs = self(inputs)
        loss = self._criterion(outputs, target_output)
        loss.backward()
        self._optimizer.step()

        return loss

    def labels_to_target_spikes(self, labels: List[int], min_spikes_count: int = 10,
                                max_spikes_count: int = 100) -> Tensor:
        """
        Create a target spike count (10 spikes for wrong label, 100 spikes for true label) in a one-hot fashion
        This approach is seen in Shrestha & Orchard (2018) https://arxiv.org/pdf/1810.08646.pdf
        Code available at https://github.com/bamsumit/slayerPytorch

        :param labels: A list of label to convert to spikes
        :param min_spikes_count: The minimum spike count of the target output (for wrong labels)
        :param max_spikes_count: The maximum spike count of the target output (for correct labels)
        :return The spike count of dimension (len(labels), network output) that represent the target output for those
         labels
        """

        target_spikes_count = min_spikes_count * torch.ones((len(labels), self.nb_classes), dtype=torch.float)
        # Use scatter ninjutsu to fill the good positions with high count (mandatory to add a dimension to labels)
        return target_spikes_count.scatter_(1, labels[:, None], max_spikes_count)

    @staticmethod
    def transform_img_to_spikes(img) -> Tensor:
        """
        Convert an image to a image to a spikes train based on the color of each pixel

        :param img: The image to convert (should be flatten)
        :return The tensor of spike time for the image, with dimension (img size, time steps)
        """
        # "Remove" the pixels associated with darker pixels (Presumably less information)
        img[img > .75] = 0
        # Re normalize [0,0.75[ => [0,1[
        img = img * 1.3333
        # Conversion to the spiking time
        # The brighter the white, the earlier the spike
        spike_time = (img * settings.absolute_duration).round_().type(torch.int64)
        # Create a value for each time step
        spikes_train = torch.zeros((len(spike_time), settings.absolute_duration + 1))
        # Write 1 at each spiking time
        spikes_train.scatter_(1, spike_time[:, None], 1)
        # Remove spike at time 0
        spikes_train = spikes_train[:, 1:]

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

    @staticmethod
    def run_spiking_layer(input_spike_train: Tensor, layer_weights: Tensor):
        """
        Here we implement a current-LIF dynamic in pytorch.
        """

        # First, we multiply the input spike train by the weights of the current layer to get the current that will be
        # added. We can calculate this beforehand because the weights are constant in the forward pass (no plasticity)
        # Equivalent to a matrix multiplication for tensors of dim > 2 using Einstein's Notation
        input_current = torch.einsum("abc,bd->adc", (input_spike_train, layer_weights))

        recorded_spikes = []  # Array of the output spikes at each time t
        membrane_potential_at_t = torch.zeros((input_spike_train.shape[0], layer_weights.shape[-1]), dtype=torch.float)
        membrane_current_at_t = torch.zeros((input_spike_train.shape[0], layer_weights.shape[-1]), dtype=torch.float)

        for t in range(settings.absolute_duration):  # For every time step
            # FIXME delta t is missing somewhere
            # Apply the leak
            # Using tau_v with euler or exact method
            membrane_potential_at_t = (1 - int(settings.delta_t) / int(settings.tau_v)) * membrane_potential_at_t
            # Using tau_i with euler or exact method
            membrane_current_at_t = (1 - int(settings.delta_t) / int(settings.tau_i)) * membrane_current_at_t

            # Select the input current at time t
            input_at_t = input_current[:, :, t]

            # Integrate the input current
            membrane_current_at_t += input_at_t

            # Integrate the input to the membrane potential
            membrane_potential_at_t += membrane_current_at_t

            # Apply the non-differentiable function
            recorded_spikes_at_t = PiecewiseSymmetric.apply(membrane_potential_at_t - settings.v_threshold)

            recorded_spikes.append(recorded_spikes_at_t)

            # Reset the spiked neurons
            membrane_potential_at_t[membrane_potential_at_t > settings.v_threshold] = 0

        # Stack over time axis (Array -> Tensor)
        return torch.stack(recorded_spikes, dim=2)


class PiecewiseSymmetric(torch.autograd.Function):
    """
    Here we implement our spiking nonlinearity which also implements the surrogate gradient.
    By subclassing torch.autograd.Function, we will be able to use all of PyTorch's autograd functionality.
    Based on Zenke & Ganguli (2018) but with a piecewise symmetric function.
    """
    alpha = 0.5

    @staticmethod
    def forward(ctx, step_input):
        """
        In the forward pass we compute a step function of the input Tensor and return it.

        :param ctx: A context object that we use to stash information which we need to later backpropagate our error
        signals.
        :param step_input: The potential of each neurons - the threshold, for a specific time step
        :return The spike train of the neurones for the next time step
        """
        ctx.save_for_backward(step_input)
        out = torch.zeros_like(step_input)
        out[step_input > 0] = 1.0  # We spike when the (potential-threshold) > 0
        return out

    @staticmethod
    def backward(ctx, grad_output):
        """
        In the backward pass we receive a Tensor we need to compute the surrogate gradient of the loss with respect to
        the input. Here we use the piecewise symmetric function.

        :param ctx: A context object that we use to recover information from the forward step.
        :param grad_output: The gradient of the previous time step for a layer (in the backward direction).
        :return The gradient of this time step.
        """
        forward_input, = ctx.saved_tensors
        grad_input = grad_output.clone()  # Clone will create a copy of the numerical value
        grad_input[forward_input <= -PiecewiseSymmetric.alpha] = 0
        grad_input[forward_input > PiecewiseSymmetric.alpha] = 0
        return grad_input
