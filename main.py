import torchvision.datasets as datasets

from networks.feed_forward import FeedForward
from run import preparation, run, clean_up


def main():
    # Prepare the environment
    preparation()

    # Load the training dataset
    train_set = datasets.MNIST(root='./tmp_data', train=True, download=True, transform=FeedForward.get_transforms())

    # Load test testing dataset
    test_set = datasets.MNIST(root='./tmp_data', train=False, download=True, transform=FeedForward.get_transforms())

    # Build the network
    network = FeedForward(input_size=len(test_set[0][0]), nb_classes=len(train_set.classes))

    # Run the training and the test
    run(train_set, test_set, network)

    # Clean up the environment, ready for a new run
    clean_up()


if __name__ == '__main__':
    main()
