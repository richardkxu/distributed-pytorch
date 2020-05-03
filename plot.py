import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


def plot_throughput(x, y, outdir):

    plt.figure(figsize=(7, 4))
    plt.plot(x, y, color='blue', marker='o')
    plt.xticks(x)
    plt.yticks(y)
    plt.xlabel('Number of GPUs')
    plt.ylabel('Training Throughput (images/sec)')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join('figures/', outdir, 'training_throughput.pdf'))
    plt.savefig(os.path.join('figures/', outdir, 'training_throughput.png'))
    plt.show()


def plot_training_time(x, y, outdir):

    plt.figure(figsize=(7, 4))
    plt.plot(x, y, color='blue', marker='o')
    plt.xticks(x)
    plt.yticks(y)
    plt.xlabel('Number of GPUs')
    plt.ylabel('Training Time (secs)')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join('figures', outdir, 'training_time.pdf'))
    plt.savefig(os.path.join('figures', outdir, 'training_time.png'))
    plt.show()


def smooth(scalars, smooth_factor):
    """
    Smoothing by exponential moving average
    :param scalars:
    :param smooth_factor:
    :return: smoothed scalars
    """
    last = scalars[0]  # First value in the plot (first timestep)
    smoothed = []
    for point in scalars:
        smoothed_val = last * smooth_factor + (1 - smooth_factor) * point  # Calculate smoothed value
        smoothed.append(smoothed_val)                        # Save it
        last = smoothed_val                                  # Anchor the last smoothed value

    return np.array(smoothed)


def plot_one_curve(ax, csv_path, color, label):
    df = pd.read_csv(csv_path)
    x, y = df['Step'], df['Value']
    y = smooth(y, 0.6)
    ax.plot(x, y, color=color, label=label)

    return ax


def plot_top1_train(paths, colors, legends, outdir):
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.set(xlabel='Epochs',
           ylabel='Top-1 Train Accuracy')
    ax.grid()

    for i in range(len(paths)):
        plot_one_curve(ax, paths[i], colors[i], legends[i])

    plt.legend(loc='lower right')
    plt.tight_layout()
    fig.savefig(os.path.join('figures', outdir, 'top1_train.pdf'))
    fig.savefig(os.path.join('figures', outdir, 'top1_train.png'))
    plt.show()


def plot_top1_val(paths, colors, legends, outdir):
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.set(xlabel='Epochs',
           ylabel='Top-1 Validation Accuracy')
    ax.grid()

    for i in range(len(paths)):
        plot_one_curve(ax, paths[i], colors[i], legends[i])

    plt.legend(loc='lower right')
    plt.tight_layout()
    fig.savefig(os.path.join('figures', outdir, 'top1_val.pdf'))
    fig.savefig(os.path.join('figures', outdir, 'top1_val.png'))
    plt.show()


def plot_top5_train(paths, colors, legends, outdir):
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.set(xlabel='Epochs',
           ylabel='Top-5 Train Accuracy')
    ax.grid()

    for i in range(len(paths)):
        plot_one_curve(ax, paths[i], colors[i], legends[i])

    plt.legend(loc='lower right')
    plt.tight_layout()
    fig.savefig(os.path.join('figures', outdir, 'top5_train.pdf'))
    fig.savefig(os.path.join('figures', outdir, 'top5_train.png'))
    plt.show()


def plot_top5_val(paths, colors, legends, outdir):
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.set(xlabel='Epochs',
           ylabel='Top-5 Validation Accuracy')
    ax.grid()

    for i in range(len(paths)):
        plot_one_curve(ax, paths[i], colors[i], legends[i])

    plt.legend(loc='lower right')
    plt.tight_layout()
    fig.savefig(os.path.join('figures', outdir, 'top5_val.pdf'))
    fig.savefig(os.path.join('figures', outdir, 'top5_val.png'))
    plt.show()


def plot_IO(bw_path, io_path, outdir):
    df = pd.read_csv(bw_path, sep=";")
    y = df['Read'] / 1.0e9
    print("Average BW: {}".format(np.sum(y[10:60]) / 50.0))
    x = np.arange(len(y))

    fig, ax1 = plt.subplots(figsize=(7, 4))
    color1 = 'tab:blue'
    ax1.set_xlabel('Minutes')
    ax1.set_ylabel('Bandwidth (GBs)', color=color1)
    ax1.plot(x, y, color=color1)
    ax1.tick_params(axis='y', labelcolor=color1)

    df = pd.read_csv(io_path, sep=";")
    z = df['Read'] / 1000.0
    print("Average IOPS: {}".format(np.sum(z[10:60]) / 50.0))
    ax2 = ax1.twinx()
    color2 = 'tab:orange'
    ax2.set_ylabel('K IOPS', color=color2)
    ax2.plot(x, z, color=color2)
    ax2.tick_params(axis='y', labelcolor=color2)

    plt.tight_layout()
    fig.savefig(os.path.join('figures', outdir, 'IO.pdf'))
    fig.savefig(os.path.join('figures', outdir, 'IO.png'))
    plt.show()


def plot_all_may():
    outdir = 'benchmark_may'
    n_gpus = np.array([4, 8, 16, 32, 64])
    throughput = np.array([2924.98, 5734.69, 10767.88, 20367.57, 37488.15])
    plot_throughput(n_gpus, throughput, outdir)

    train_time = np.array([40908.36, 21123.88, 11291.56, 6027.09, 3378.84])
    plot_training_time(n_gpus, train_time, outdir)

    colors = ['red', 'blue', 'purple', 'brown', 'gray']
    legends = ['gpux4', 'gpux8', 'gpux16', 'gpux32', 'gpux64']
    top1_train_paths = ['Top1_train/run-May01_17-14-54_hal08_resnet50_gpux4_b224_cpu20_optO2-tag-Top1_train.csv',
                        'Top1_train/run-May01_10-28-28_hal06_resnet50_gpux8_b256_cpu20_optO2-tag-Top1_train.csv',
                        'Top1_train/run-May01_12-59-11_hal03_resnet50_gpux16_b256_cpu20_optO2-tag-Top1_train.csv',
                        'Top1_train/run-Apr30_23-29-32_hal01_resnet50_gpux32_b256_cpu20_optO2-tag-Top1_train.csv',
                        'Top1_train/run-May01_08-52-56_hal01_resnet50_gpux64_b256_cpu20_optO2-tag-Top1_train.csv'
                        ]
    plot_top1_train(top1_train_paths, colors, legends, outdir)

    top1_val_paths = ['Top1_val/run-May01_17-14-54_hal08_resnet50_gpux4_b224_cpu20_optO2-tag-Top1_val.csv',
                      'Top1_val/run-May01_10-28-28_hal06_resnet50_gpux8_b256_cpu20_optO2-tag-Top1_val.csv',
                      'Top1_val/run-May01_12-59-11_hal03_resnet50_gpux16_b256_cpu20_optO2-tag-Top1_val.csv',
                      'Top1_val/run-Apr30_23-29-32_hal01_resnet50_gpux32_b256_cpu20_optO2-tag-Top1_val.csv',
                      'Top1_val/run-May01_08-52-56_hal01_resnet50_gpux64_b256_cpu20_optO2-tag-Top1_val.csv'
                      ]
    plot_top1_val(top1_val_paths, colors, legends, outdir)

    top5_train_paths = ['Top5_train/run-May01_17-14-54_hal08_resnet50_gpux4_b224_cpu20_optO2-tag-Top5_train.csv',
                        'Top5_train/run-May01_10-28-28_hal06_resnet50_gpux8_b256_cpu20_optO2-tag-Top5_train.csv',
                        'Top5_train/run-May01_12-59-11_hal03_resnet50_gpux16_b256_cpu20_optO2-tag-Top5_train.csv',
                        'Top5_train/run-Apr30_23-29-32_hal01_resnet50_gpux32_b256_cpu20_optO2-tag-Top5_train.csv',
                        'Top5_train/run-May01_08-52-56_hal01_resnet50_gpux64_b256_cpu20_optO2-tag-Top5_train.csv'
                        ]
    plot_top5_train(top5_train_paths, colors, legends, outdir)

    top5_val_paths = ['Top5_val/run-May01_17-14-54_hal08_resnet50_gpux4_b224_cpu20_optO2-tag-Top5_val.csv',
                      'Top5_val/run-May01_10-28-28_hal06_resnet50_gpux8_b256_cpu20_optO2-tag-Top5_val.csv',
                      'Top5_val/run-May01_12-59-11_hal03_resnet50_gpux16_b256_cpu20_optO2-tag-Top5_val.csv',
                      'Top5_val/run-Apr30_23-29-32_hal01_resnet50_gpux32_b256_cpu20_optO2-tag-Top5_val.csv',
                      'Top5_val/run-May01_08-52-56_hal01_resnet50_gpux64_b256_cpu20_optO2-tag-Top5_val.csv'
                      ]
    plot_top5_val(top5_val_paths, colors, legends, outdir)

    plot_IO('IO/bandwidth_gpux64_cpu20_may01.csv', 'IO/iops_gpux64_cpu20_may01.csv', outdir)

    return


def plot_all_feb():
    outdir = 'benchmark_feb'
    n_gpus = np.array([2, 4, 8, 16, 32, 64])
    throughput = np.array([1582.04, 3004.51, 5805.14, 11273.57, 19496.10, 33080.03])
    plot_throughput(n_gpus, throughput, outdir)

    train_time = np.array([74211.11, 39536, 20904.01, 10969.42, 6652.91, 4071.31])
    plot_training_time(n_gpus, train_time, outdir)

    colors = ['orange', 'red', 'blue', 'purple', 'brown', 'gray']
    legends = ['gpux2', 'gpux4', 'gpux8', 'gpux16', 'gpux32', 'gpux64']
    top1_train_paths = ['Top1_train/run-Feb09_14-20-42_hal14_resnet50_gpux2_b208_cpu20_optO2-tag-Top1_train.csv',
                        'Top1_train/run-Feb09_14-22-11_hal13_resnet50_gpux4_b208_cpu20_optO2-tag-Top1_train.csv',
                        'Top1_train/run-Feb08_13-47-09_hal11_resnet50_gpux8_b208_cpu20_optO2-tag-Top1_train.csv',
                        'Top1_train/run-Feb09_09-21-23_hal13_resnet50_gpux16_b208_cpu20_optO2-tag-Top1_train.csv',
                        'Top1_train/run-Feb12_23-28-54_hal01_resnet50_gpux32_b208_cpu20_optO2-tag-Top1_train.csv',
                        'Top1_train/run-Feb12_21-54-28_hal01_resnet50_gpux64_b208_cpu20_optO2-tag-Top1_train.csv'
                        ]
    plot_top1_train(top1_train_paths, colors, legends, outdir)

    top1_val_paths = ['Top1_val/run-Feb09_14-20-42_hal14_resnet50_gpux2_b208_cpu20_optO2-tag-Top1_val.csv',
                      'Top1_val/run-Feb09_14-22-11_hal13_resnet50_gpux4_b208_cpu20_optO2-tag-Top1_val.csv',
                      'Top1_val/run-Feb08_13-47-09_hal11_resnet50_gpux8_b208_cpu20_optO2-tag-Top1_val.csv',
                      'Top1_val/run-Feb09_09-21-23_hal13_resnet50_gpux16_b208_cpu20_optO2-tag-Top1_val.csv',
                      'Top1_val/run-Feb12_23-28-54_hal01_resnet50_gpux32_b208_cpu20_optO2-tag-Top1_val.csv',
                      'Top1_val/run-Feb12_21-54-28_hal01_resnet50_gpux64_b208_cpu20_optO2-tag-Top1_val.csv'
                      ]
    plot_top1_val(top1_val_paths, colors, legends, outdir)

    top5_train_paths = ['Top5_train/run-Feb09_14-20-42_hal14_resnet50_gpux2_b208_cpu20_optO2-tag-Top5_train.csv',
                        'Top5_train/run-Feb09_14-22-11_hal13_resnet50_gpux4_b208_cpu20_optO2-tag-Top5_train.csv',
                        'Top5_train/run-Feb08_13-47-09_hal11_resnet50_gpux8_b208_cpu20_optO2-tag-Top5_train.csv',
                        'Top5_train/run-Feb09_09-21-23_hal13_resnet50_gpux16_b208_cpu20_optO2-tag-Top5_train.csv',
                        'Top5_train/run-Feb12_23-28-54_hal01_resnet50_gpux32_b208_cpu20_optO2-tag-Top5_train.csv',
                        'Top5_train/run-Feb12_21-54-28_hal01_resnet50_gpux64_b208_cpu20_optO2-tag-Top5_train.csv'
                        ]
    plot_top5_train(top5_train_paths, colors, legends, outdir)

    top5_val_paths = ['Top5_val/run-Feb09_14-20-42_hal14_resnet50_gpux2_b208_cpu20_optO2-tag-Top5_val.csv',
                      'Top5_val/run-Feb09_14-22-11_hal13_resnet50_gpux4_b208_cpu20_optO2-tag-Top5_val.csv',
                      'Top5_val/run-Feb08_13-47-09_hal11_resnet50_gpux8_b208_cpu20_optO2-tag-Top5_val.csv',
                      'Top5_val/run-Feb09_09-21-23_hal13_resnet50_gpux16_b208_cpu20_optO2-tag-Top5_val.csv',
                      'Top5_val/run-Feb12_23-28-54_hal01_resnet50_gpux32_b208_cpu20_optO2-tag-Top5_val.csv',
                      'Top5_val/run-Feb12_21-54-28_hal01_resnet50_gpux64_b208_cpu20_optO2-tag-Top5_val.csv'
                      ]
    plot_top5_val(top5_val_paths, colors, legends, outdir)

    plot_IO('IO/bandwidth_gpux64_feb.csv', 'IO/iops_gpux64_feb.csv', outdir)

    return


if __name__ == '__main__':
    plot_all_may()

