import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from torch.utils.data import Dataset


class MockClassificationDataset(Dataset):
    """
    Randomly generate a datasets of 2 dimensional points grouped in N classes.
    """

    def __init__(self, nb_classes: int = 2, data_point_per_class: int = 100):
        """
        Create a random dataset.

        :param nb_classes: Number of class
        :param data_point_per_class: Number of data point per class
        """
        self._nb_classes = nb_classes
        self._data_point_per_class = data_point_per_class

        self._features = np.zeros((nb_classes * data_point_per_class, 2), dtype=np.single)
        self._labels = np.zeros(nb_classes * data_point_per_class, dtype=np.int_)

        # Generate points for each classes with normal distribution.
        # Every class have a different center.
        for cl in range(nb_classes):
            index_start = cl * data_point_per_class
            index_end = cl * data_point_per_class + data_point_per_class

            self._features[index_start:index_end] = np.random.normal(loc=cl * 2, scale=1,
                                                                     size=(data_point_per_class, 2))
            self._labels[index_start:index_end] = cl

    def __len__(self):
        return len(self._labels)

    def __getitem__(self, index):
        return self._features[index], self._labels[index]

    def show_plot(self) -> None:
        """
        Create a plot that represent the dataset and show it.
        """
        sns.scatterplot(x=self._features[:, 0],
                        y=self._features[:, 1],
                        hue=self._labels,
                        markers=True)
        plt.title(f'Data from {self._nb_classes} classes')
        plt.show()
