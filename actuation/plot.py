import csv
import argparse
import matplotlib
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal


class Plot():
    def __init__(self, args):
        self.file = args.file
        self.path = args.path
        self.smoothing = args.smoothing

    def __call__(self):
        with open(self.path + '/' + self.file, 'r', newline='') as file:
            reader = csv.reader(file)
            data = list()
            for row in reader:
                data.append(row)

            data = np.asarray(data, dtype=np.float32)

            data = np.rot90(data)

            if self.smoothing:
                data = signal.savgol_filter(data, 23, 1)

            for dataset in data:
                plt.plot(dataset)
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.show()


if __name__ == '__main__':
    def str2bool(v):
        return v.lower() in ("yes", "true", "t", "1")

    parser = argparse.ArgumentParser()
    parser.add_argument('--file', type=str)
    parser.add_argument('--path', default='../data/checkpoints', type=str)
    parser.add_argument('--smoothing', default=True, type=str2bool)

    args = parser.parse_args()

    plot = Plot(args)
    plot()
