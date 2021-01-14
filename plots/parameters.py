import matplotlib.pyplot as plt
import seaborn as sns
from torch.nn import Module
from torch.nn.utils import parameters_to_vector

from utils.output import save_plot

_parameters_history = {}


def store_parameters(network: Module, labels: str = ''):
    _parameters_history[labels] = parameters_to_vector(network.parameters()).detach()


def parameters_distribution(network: Module, title_context: str = ''):
    parameters = parameters_to_vector(network.parameters()).detach()
    sns.histplot(parameters, bins=20)
    plt.title(f'Network\'s parameters distribution {title_context}')
    plt.xlabel('Parameter values')
    plt.ylabel('Count')
    save_plot('parameters_distribution ' + title_context)


def parameters_distribution_group(network: Module, title_context: str = ''):
    raise NotImplemented
    sns.displot(
        x=_parameters_history.values(), kind="hist", height=4, aspect=.7,
    )
    plt.title(f'Network\'s parameters distribution {title_context}')
    plt.xlabel('Parameter values')
    plt.ylabel('Count')
    save_plot()
