import random
import tensorflow as tf
import pandas as pd
import numpy as np
from multiprocessing import Pool


class ProcessKaggle():
    def __init__(self, rawDir, proDir, batchSize, threadNum, preFetch, valSize):
        self.rawDir = rawDir
        self.proDir = proDir
        self.batchSize = batchSize
        self.threadNum = threadNum
        self.preFetch = preFetch
        self.valSize = valSize

    def _dataPoint(self, row):
        return row[:30], np.array([int(item) for item in row[30].split()]).reshape((96, 96))

    def _buildData(self):
        fileName = '/training.csv'
        df = pd.read_csv(self.rawDir + '/' + fileName)

        data = df.to_numpy()
        print("data shape")
        print(data.shape)

        with Pool(processes=self.threadNum) as pool:
            tuple = pool.map(self._dataPoint, data)

        images, points = zip(*tuple)
        images = np.asarray(images).reshape((len(images), 96, 96, 1))
        points = np.asarray(points).reshape((len(points), 30))


        np.save(self.proDir + '/images.npy', images)
        np.save(self.proDir + '/points.npy', points)

        return images, points

    def loadData(self):
        try:
            images = np.load(self.proDir + '/images.npy')
            points = np.load(self.proDir + '/points.npy')
        except FileNotFoundError:
            images, points = self._buildData()

        zipped = list(zip(images, points))
        random.shuffle(zipped)
        images, points = zip(*zipped)
        images = np.asarray(images)
        points = np.asarray(points)

        dataset = tf.data.Dataset.from_tensor_slices((images, points))
        dataset = dataset.batch(self.batchSize)
        dataset = dataset.prefetch(self.preFetch)
        valDataset = dataset.take(self.valSize)
        trainDataset = dataset.take(images.shape[0] - self.valSize)

        return trainDataset, valDataset
