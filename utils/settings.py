from dataclasses import dataclass, asdict
from typing import Union

import configargparse

from utils.logger import logger


@dataclass(init=True, repr=True, eq=True, order=False, unsafe_hash=False, frozen=False)
class Settings:
    """
    Storing all settings for this program with default values.
    Setting are loaded from (last override first):
        - default values (in this file)
        - local file (default path: ./settings.yaml)
        - environment variables
        - arguments of the command line (with "--" in front)
    """

    # Name of the run to save the result ('tmp' for temporary files)
    run_name: str = ''

    # Output configuration
    logger_output_level: Union[str, int] = 'INFO'
    show_images: bool = False

    # Resolution reduction
    min_value: float = -0.5
    max_value: float = 0.5
    inaccuracy_value: float = 0.25

    # Training settings
    batch_size: int = 4
    nb_epoch: int = 4
    learning_rate: float = 0.001

    # Network configuration
    size_hidden_1: int = 50
    size_hidden_2: int = 50

    def validate(self):
        """
        Validate settings.
        """
        # TODO check if run_name have valid character for a file
        assert self.logger_output_level in ['CRITICAL', 'FATAL', 'ERROR', 'WARN', 'WARNING', 'INFO', 'DEBUG',
                                            'NOTSET'] or isinstance(self.logger_output_level, int), 'Invalid log level'

        assert self.batch_size > 0, 'Batch size should be a positive integer'
        assert self.nb_epoch > 0, 'Number of epoch should be at least 1'

    def _load_file_and_cmd(self) -> None:
        """
        Load settings from local file and arguments of the command line.
        """

        p = configargparse.get_argument_parser(default_config_files=['./settings.yaml'])

        p.add_argument('-s', '--settings', required=False, is_config_file=True,
                       help='path to custom configuration file')

        for name, value in asdict(self).items():
            p.add_argument(f'--{name}', required=False, type=type(value))

        # TODO deal with unknown arguments with a warning
        for name, value in vars(p.parse_args()).items():
            if name == 'settings':
                continue
            if value is not None:
                # Directly set the value to bypass the "__setattr__" function
                self.__dict__[name] = value

        self.validate()

    def __post_init__(self):
        """
        Called after the init of this object.
        """
        self._load_file_and_cmd()

    def __setattr__(self, name, value) -> None:
        """
        Set an attribute and valide the new value.

        :param name: The name of the attribut
        :param value: The value of the attribut
        """
        logger.info(f'Setting "{name}" changed from "{getattr(self, name)}" to "{value}".')
        self.__dict__[name] = value
        self.validate()

    def __delattr__(self, name):
        raise AttributeError('Removing a setting is forbidden for the sake of consistency.')

    def __str__(self) -> str:
        """
        :return: Human readable description of the settings.
        """
        return 'Settings:\n\t' + \
               '\n\t'.join([f'{name}: {str(value)}' for name, value in asdict(self).items()])


# Singleton setting object
settings = Settings()
