import numpy as np
import matplotlib.pyplot as plt


def plot_throughput():

    x = np.array([2, 4, 8, 16, 32, 64])
    y = np.array([1582.04, 3004.51, 5805.14, 11273.57, 19496.10, 33080.03])
    # z = np.array([1, 2, 4, 8, 16])*3004.51

    # linear
    plt.plot(x, y, color='blue', marker='o')
    # plt.plot(x, z, color='gray', linestyle='dashed', marker='o')
    plt.xticks(x)
    plt.yticks(y)
    plt.xlabel('Number of GPUs')
    plt.ylabel('Global Throughput (images/sec)')
    plt.title('Training Throughput')
    plt.grid(True)
    plt.savefig('figures/training_throughput.pdf')


def plot_training_time():
    x = np.array([2, 4, 8, 16, 32, 64])
    y = np.array([74211.11, 39536, 20904.01, 10969.42, 6652.91, 4071.31])

    # linear
    plt.plot(x, y, color='blue', marker='o')
    plt.xticks(x)
    plt.yticks(y)
    plt.xlabel('Number of GPUs')
    plt.ylabel('Time to Solution (secs)')
    plt.title('Training Time')
    plt.grid(True)
    # plt.show()
    plt.savefig('figures/training_time.pdf')


if __name__ == '__main__':
    plot_throughput()
    # plot_training_time()
