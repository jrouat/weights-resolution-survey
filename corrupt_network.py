import torch
from torch.nn import Module


def reduce_resolution(network: Module, min_value: float, max_value: float, delta: float) -> None:
    """
    Reduce the resolution of the parameters of the network in place.

    :param network The network the process
    :param min_value The minimal value of the parameters
    :param max_value The maximum value of the parameters
    :param delta The bigger step discernible for the new resolution
    """
    for param in network.parameters():
        # Noise +/- (delta / 2)
        noise = (torch.rand(param.shape) * delta) - (delta / 2)
        # Add noise and restrict the range
        param.data = torch.clip(param.data + noise, min_value, max_value)
