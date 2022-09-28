"""
Script used to plot the details of the training run, which is specified in the log files
"""

from model.monitor_utils import plot_detailed_results
import os

if __name__ == '__main__':

    log_files = ['file_path1/progress.csv',
                 'file_path2/progress.csv']

    algos_names = ['trpo', 'trpo']

    name_exps = ['Experiment 1', 'Experiment 2']

    save_dir = 'save_dir_path'

    os.makedirs(save_dir, exist_ok=True)
    plot_detailed_results(log_files, save_dir, algos_names, name_exps, window_size=20)
