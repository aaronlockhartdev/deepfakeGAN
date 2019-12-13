import tensorflow as tf
import numpy as np
import cv2
import os
import random
from multiprocessing import Pool
from tensorflow.data import Dataset


class ProcessVids():

    def __init__(self, rawDir, proDir, batchSize, valSize, threadNum, preFetch):
        self.rawDir = rawDir
        self.proDir = proDir
        self.batchSize = batchSize
        self.valSize = valSize
        self.threadNum = threadNum
        self.preFetch = preFetch
        self.fileList = os.listdir(rawDir)

    def loadData(self):
        try:
            proVidData = np.load(self.proDir + '/processed_data.npy')
        except FileNotFoundError:
            proVidData = self._buildData()

        inputData = list()
        outputData = list()
        for v0 in proVidData:
            for v1 in proVidData:
                if len(v0) < len(v1):
                    inputData.append(v0)
                    outputData.append(v1[:len(v0)])
                elif len(v1) < len(v0):
                    inputData.append(v0[:len(v1)])
                    outputData.append(v1)
                else:
                    inputData.append(v0)
                    outputData.append(v1)

        inputData = np.asarray(inputData)
        outputData = np.asarray(outputData)

        inputData = np.reshape(
            inputData, (inputData.shape[0] * inputData.shape[1], inputData.shape[2], inputData.shape[3]))
        outputData = np.reshape(
            outputData, (outputData.shape[0] * outputData.shape[1], outputData.shape[2], outputData.shape[3]))

        zipped = list(zip(inputData, outputData))
        random.shuffle(zipped)
        inputData, outputData = zip(*zipped)
        inputData = np.asarray(inputData)
        outputData = np.asarray(outputData)

        dataset = Dataset.from_tensor_slices((inputData, outputData))
        dataset = dataset.batch(self.batchSize)
        dataset = dataset.prefetch(self.preFetch)
        valDataset = dataset.take(self.valSize)
        trainDataset = dataset.take(len(dataList) - self.valSize)

        return trainDataset, valDataset

    def _buildData(self):

        for f in self.fileList:
            f = rawDir + '/' + f

        with Pool(processes=self.threadNum) as pool:
            vidData = pool.map(self._getFrames, self.fileList)
            proVidData = list()
            for v in vidData:
                proVidData.append(pool.map(self._procData, v))

        proVidData = np.asarray(proVidData)

        np.save(proDir + '/processed_data.npy', proVidData)

        return proVidData

    def _getFrames(self, fileName):
        cap = cv2.VideoCapture(fileName)
        frameCount = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frameWidth = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frameHeight = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        buf = np.empty((frameCount, frameHeight, frameWidth, 3), np.dtype('uint8'))

        fc = 0
        ret = True

        while (fc < frameCount and ret):
            ret, buf[fc] = cap.read()
            fc += 1

        cap.release()

        return buf

    def _procData(self, frame):
        faceCascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
        faces = faceCascade.detectMultiScale(frame, 1.1, 4)
        if len(faces) == 0:
            return None
        result = cv2.resize(faces[0], (200, 200))
        result /= 255
        return result
