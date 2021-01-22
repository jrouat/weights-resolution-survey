import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from utils.output import load_runs, set_plot_style

if __name__ == '__main__':
    # Set plot style
    set_plot_style()

    # ========================== Inaccuracy Value ==========================

    # Load selected runs files
    data = load_runs('inaccuracy_value*')

    data['nb_states'] = (data['settings.max_value'] - data['settings.min_value']) / data['settings.inaccuracy_value']
    data['bits'] = np.log2(data['nb_states'])

    sns.lineplot(data=data, x='bits', y='results.accuracy_low_resolution', hue='network_info.name')
    plt.title('Evolution of the testing accuracy depending of\nweights resolution')
    plt.xlabel('Resolution (bits)')
    plt.ylabel('Accuracy')
    plt.show(block=False)

    # nb epoch

    # hidden size / nb parameters

    # min / max
