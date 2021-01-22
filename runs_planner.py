from pathlib import Path

import numpy as np

from main import main
from utils.logger import logger
from utils.output import OUT_DIR
from utils.settings import settings

if __name__ == '__main__':
    planning = {
        'max_value': {'name': 'weight_limits_8_ff',
                      'range': map(float, np.arange(0.05, 2, 0.05)),
                      'pre-trained': 'baseline-feedforward',
                      }
    }

    # Iterate through all planning settings and values
    for setting, plan in planning.items():
        for i, value in enumerate(plan['range'], start=1):
            logger.info(f'Start {setting} with value {value}')
            # Change the settings
            settings.run_name = f"{plan['name']}-{i:03}"
            setattr(settings, setting, value)
            setattr(settings, 'min_value', -value)
            setattr(settings, 'inaccuracy_value', value / 4)
            # Set pre trained if set
            if 'pre-trained' in plan:
                settings.trained_network_cache_path = str(Path(OUT_DIR, plan['pre-trained'], 'trained_network.p'))
            # Start the run
            main()
