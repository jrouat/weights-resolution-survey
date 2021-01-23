import torchvision.datasets as datasets

from networks.snn import SNN
from run import preparation, run, clean_up
from utils.logger import logger


def main():
    # Prepare the environment
    preparation()

    # Catch and log every exception during the runtime
    # noinspection PyBroadException
    try:
        # Load the training dataset
        train_set = datasets.MNIST(root='./tmp_data', train=True, download=True, transform=SNN.get_transforms())

        # Load test testing dataset
        test_set = datasets.MNIST(root='./tmp_data', train=False, download=True, transform=SNN.get_transforms())

        # Build the network
        network = SNN(input_size=len(test_set[0][0]), nb_classes=len(train_set.classes))

        # Run the training and the test
        run(train_set, test_set, network)
    except KeyboardInterrupt:
        logger.error('Run interrupted by the user.')
        raise  # Let it go to stop the runs planner if needed
    except Exception:
        logger.critical('Run interrupted by an unexpected error.', exc_info=True)
    finally:
        # Clean up the environment, ready for a new run
        clean_up()


if __name__ == '__main__':
    main()
