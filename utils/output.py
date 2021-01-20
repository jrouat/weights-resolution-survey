from dataclasses import asdict
from pathlib import Path
from typing import Any, Union

import matplotlib.pyplot as plt
import torch
from torch.nn import Module

from utils.logger import logger
from utils.settings import settings

OUT_DIR = './out'


def init_out_directory() -> None:
    """
    Prepare the output directory.
    """
    run_dir = Path(OUT_DIR, settings.run_name)
    img_dir = run_dir / 'img'

    # If the keyword 'tmp' is used as run name, then remove the previous files
    if settings.run_name == 'tmp':
        logger.warning(f'Using temporary directory to save this run results.')
        if run_dir.exists():
            logger.warning(f'Previous temporary files removed: {run_dir}')
            # Remove text files
            (run_dir / 'settings.yaml').unlink(missing_ok=True)
            (run_dir / 'results.yaml').unlink(missing_ok=True)
            (run_dir / 'network_info.yaml').unlink(missing_ok=True)
            (run_dir / 'run.log').unlink(missing_ok=True)

            # Remove images
            if img_dir.is_dir():
                # Remove png images files
                for png_file in img_dir.glob('*.png'):
                    png_file.unlink()
                img_dir.rmdir()

            # Remove saved networks
            for p_file in run_dir.glob('*.p'):
                p_file.unlink()

            # Remove tmp directory
            run_dir.rmdir()

    # Create the directories
    img_dir.mkdir(parents=True)
    logger.info(f'Output directory created: {run_dir}')

    # Init the logger file
    if settings.logger_file_enable:
        logger.enable_log_file(file_path=(run_dir / 'run.log'), file_log_level=settings.logger_file_level)

    parameter_file = run_dir / 'settings.yaml'
    with open(parameter_file, 'w+') as f:
        f.write('\n'.join([f'{name}: {str(value)}' for name, value in asdict(settings).items()]))

    logger.debug(f'Parameters saved in {parameter_file}')


def save_network_info(network_metrics: dict) -> None:
    """
    Save metrics information in a file in the run directory.

    :param network_metrics: The dictionary of metrics with their values.
    """
    run_dir = Path(OUT_DIR, settings.run_name)
    network_info_file = run_dir / 'network_info.yaml'
    with open(network_info_file, 'w+') as f:
        f.write('\n'.join([f'{name}: {str(value)}' for name, value in network_metrics.items()]))

    logger.debug(f'Network info saved in {network_info_file}')


def save_results(**results: Any) -> None:
    """
    Write a new line in the result file.

    :param results: Dictionary of labels and values, could be anything that implement __str__.
    """
    results_path = Path(OUT_DIR, settings.run_name, 'results.yaml')

    # Append to the file, create it if necessary
    with open(results_path, 'a') as f:
        for label, value in results.items():
            f.write(f'{label}: {str(value)}\n')

    logger.debug(f'{len(results)} result(s) saved in {results_path}')


def save_plot(file_name: str) -> None:
    """
    Save a plot image in the directory
    """
    save_path = Path(OUT_DIR, settings.run_name, 'img', f'{file_name}.png')
    plt.savefig(save_path)
    logger.debug(f'Plot saved in {save_path}')

    # Plot image or close it
    plt.show(block=False) if settings.show_images else plt.close()


def save_network(network: Module, file_name: str = 'network') -> None:
    """
    Save a full description of the network parameters and states.

    :param network: The network to save
    :param file_name: The name of the destination file (without the extension)
    """
    cache_path = Path(OUT_DIR, settings.run_name, file_name + '.p')
    torch.save(network.state_dict(), cache_path)
    logger.info(f'Network saved ({cache_path})')


def load_network(network: Module, file_path: Union[str, Path]) -> bool:
    """
    Load a full description of the network parameters and states from a previous save file.

    :param network: The network to load into (in place)
    :param file_path: The path to the file to load
    :return: True if the file exist and is loaded, False if the file is not found.
    """
    cache_path = Path(file_path) if isinstance(file_path, str) else file_path
    if cache_path.is_file():
        network.load_state_dict(torch.load(cache_path))
        logger.info(f'Network loaded ({cache_path})')
        return True
    logger.warning(f'Network cache not found in "{cache_path}"')
    return False
