from pathlib import Path

import numpy as np

from main import main
from utils.logger import logger
from utils.output import OUT_DIR
from utils.settings import settings

if __name__ == '__main__':
    planning = {
        'inaccuracy_value': {'name': 'inaccuracy_value',
                             'range': map(float, np.arange(0., 0.5, 0.025)),
                             'pre-trained': 'ref'}
    }

    # Iterate through all planning settings and values
    for setting, plan in planning.items():
        for i, value in enumerate(plan['range'], start=1):
            logger.info(f'Start {setting} with value {value}')
            # Change the settings
            settings.run_name = f"{plan['name']}-{i:03}"
            setattr(settings, setting, value)
            # Set pre trained if set
            if 'pre-trained' in plan:
                settings.trained_network_cache_path = str(Path(OUT_DIR, plan['pre-trained'], 'trained_network.p'))
            # Start the run
            main()
