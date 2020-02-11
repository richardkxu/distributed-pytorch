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
    x = np.array([2, 4, 8, 16])
    t = np.array([74211.11, 39536, 20904, 10969.42])

    # linear
    plt.plot(x, t, 'o-')
    plt.xlabel('Number of GPUs')
    plt.ylabel('Time in sec')
    plt.title('Training Time')
    plt.grid(True)
    plt.show()


if __name__ == '__main__':
    plot_throughput()
    plot_time_elapse()