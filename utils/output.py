import glob
from dataclasses import asdict
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt

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

            # Remove images
            if img_dir.is_dir():
                # Remove png images files
                for png_file in glob.glob(str(img_dir / '*.png')):
                    Path(png_file).unlink()
                img_dir.rmdir()

            # Remove tmp directory
            run_dir.rmdir()

    img_dir.mkdir(parents=True)

    logger.info(f'Output directory created: {run_dir}')

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

    :param results dictionary of labels and values.
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
