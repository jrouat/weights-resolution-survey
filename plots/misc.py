from typing import List

import matplotlib.pyplot as plt
import seaborn as sns


def plot_losses(loss_evolution: List[float]) -> None:
    sns.relplot(data=loss_evolution, kind='line')
    plt.title('Loss evolution')
    plt.xlabel('Batch number')
    plt.ylabel('Loss (Cross Entropy)')
    plt.show()
