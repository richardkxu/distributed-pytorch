import numpy as np
import matplotlib.pyplot as plt


def plot_throughput():

    x = np.array([2, 4, 8, 16])
    y = np.array([1582, 3005, 5793, 11298])

    # linear
    plt.plot(x, y, 'o-')
    plt.xlabel('Number of GPUs')
    plt.ylabel('Images per sec')
    plt.title('Training Throughput')
    plt.grid(True)
    plt.show()


def plot_time_elapse():
    x = np.array([2, 4, 8, 16, 32, 64])
    t = np.array([74211.11, 39536, 20904.01, 10969.42, 6652.91, 4071.31])

    # linear
    plt.plot(x, t, 'o-')
    plt.xticks(x)
    plt.yticks(t)
    plt.xlabel('Number of GPUs')
    plt.ylabel('Time Elapse (secs)')
    plt.title('Training Time')
    plt.grid(True)
    plt.show()


if __name__ == '__main__':
    plot_time_elapse()