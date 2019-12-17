import random
import tensorflow as tf
import pandas as pd
import numpy as np
from multiprocessing import Pool
from sklearn.impute import SimpleImputer


class ProcessKaggle():
    def __init__(self, rawDir, proDir, batchSize, threadNum, preFetch, valSize):
        self.rawDir = rawDir
        self.proDir = proDir
        self.batchSize = batchSize
        self.threadNum = threadNum
        self.preFetch = preFetch
        self.valSize = valSize

    def _split(self, string):
        return np.array(string.split(), dtype=np.float32).reshape((96, 96, 1))

    def _buildData(self):
        fileName = '/training.csv'
        df = pd.read_csv(self.rawDir + '/' + fileName)
        # df = df.dropna()

        points = np.array(df.iloc[:, : 30], dtype=np.float32)
        images = np.array(df.iloc[:, 30])

        with Pool(processes=self.threadNum) as pool:
            images = np.array(pool.map(self._split, images))

        imputer = SimpleImputer()

        points = imputer.fit_transform(points)

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
