import matplotlib.pyplot as plt
import seaborn as sns

from utils.output import load_runs, set_plot_style

if __name__ == '__main__':
    # Set plot style
    set_plot_style()
    # Load selected runs' files
    data = load_runs('*')

    # Evolution of the accuracy depending of the number of parameters
    sns.lineplot(data=data, x='network_info.total_params', y='results.accuracy_ideal')
    sns.lineplot(data=data, x='network_info.total_params', y='results.accuracy_low_resolution', legend='full')
    plt.title('Evolution of the accuracy depending of the number\nof parameters')
    plt.xlabel('Number of parameters')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show(block=False)
