from datasets.mock_classification_dataset import MockClassificationDataset
from networks.simple_classifier import SimpleClassifier
from run import preparation, run

if __name__ == '__main__':
    # Prepare the environment
    preparation()

    # Load the training dataset
    train_set = MockClassificationDataset(nb_classes=4, data_point_per_class=200)
    train_set.show_plot()  # Plot and show the data

    # Load test testing dataset
    test_set = MockClassificationDataset(nb_classes=4, data_point_per_class=50)

    # Build the network
    net = SimpleClassifier(input_size=2, nb_classes=4)

    # Run the training and the test
    run(train_set, test_set, net)
