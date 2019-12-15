import os
import csv
import sys
import argparse
import numpy as np
import tensorflow as tf
from tensorflow import GradientTape
from tensorflow_addons.optimizers import AdamW
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import MSE


class TrainPose():
    def __init__(self, args):
        # initialize tensorflow
        gpu = tf.config.experimental.list_physical_devices('GPU')[0]
        tf.config.experimental.set_memory_growth(gpu, True)
        sys.setrecursionlimit(10000)

        print("Eager execution is " + str(tf.executing_eagerly()))
        print("Tensorflow version " + str(tf.__version__))
        # import classes
        sys.path.append("..")
        from models.feature_pose import FeaturePose
        from preprocessing.process_kaggle import ProcessKaggle
        # initialize training variables
        self.numEpochs = args.numEpochs

        # self.optimizer = AdamW(1e-2 * args.lr, learning_rate=args.lr)
        self.optimizer = Adam(args.lr)

        self.loss = None

        self.restore = args.restore

        # initialize preprocessor and build/load data
        preprocessor = ProcessKaggle(args.rawDir, args.proDir,
                                     args.batchSize, args.threadNum,
                                     args.preFetch, args.valSize)

        self.trainData, self.valData = preprocessor.loadData()

        # initialize models

        self.model = FeaturePose()

        # initialize checkpoints and managers
        if not self.restore:
            for f in os.listdir('../data/checkpoints/feature_pose'):
                os.remove('../data/checkpoints/feature_pose/' + f)
            if os.path.isfile('../data/checkpoints/feature_pose_losses.csv'):
                os.remove('../data/checkpoints/feature_pose_losses.csv')

        self.checkpoint = tf.train.Checkpoint(optimizer=self.optimizer,
                                              model=self.model)
        self.manager = tf.train.CheckpointManager(self.checkpoint,
                                                  directory='../data/checkpoints/feature_pose',
                                                  max_to_keep=5, keep_checkpoint_every_n_hours=2)

    def _loss(self, x, y):
        return MSE(y, x)

    def _logTrain(self, epoch, batch):
        msg = 'Training Batch {} in Epoch {}, Model Loss: {}'
        print(msg.format(batch + 1, epoch + 1, self.loss), end='\r')

    def _logVal(self, epoch, batch):
        msg = 'Validating Batch {} in Epoch {}, Model Loss: {}'
        print(msg.format(batch + 1, epoch + 1, self.loss), end='\r')

    def _save(self):
        self.manager.save()
        with open('../data/checkpoints/feature_pose_losses.csv', 'a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([self.lossTrainAverage, self.lossValAverage])

    @tf.function
    def _validate(self, input, output):
        predicted = self.model.call(input)
        loss = self._loss(predicted, output)

        return tf.math.reduce_mean(loss)

    @tf.function
    def _update(self, input, output):
        with GradientTape() as tape:
            predicted = self.model.call(input, training=True)
            loss = self._loss(predicted, output)

            gradients = tape.gradient(loss, self.model.trainable_variables)
            self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
            return tf.math.reduce_mean(loss)

    def train(self):
        if self.restore:
            self.checkpoint.restore(self.manager.latest_checkpoint)

        for epoch in range(self.numEpochs):
            self.lossTrainAverage = list()
            self.lossValAverage = list()
            for ind, (input, output) in self.trainData.enumerate():
                self.loss = self._update(input, output).numpy()
                self.lossTrainAverage.append(self.loss)
                self._logTrain(epoch, ind)
            self.trainData = self.trainData.shuffle(10000)
            for ind, (input, output) in self.valData.enumerate():
                self.loss = self._validate(input, output).numpy()
                self.lossValAverage.append(self.loss)
                self._logVal(epoch, ind)

            self.lossTrainAverage = np.average(np.asarray(self.lossTrainAverage))
            self.lossValAverage = np.average(np.asarray(self.lossValAverage))
            print('Epoch {} of {} complete with training loss of {} and validation loss of {}'.format(
                epoch + 1, self.numEpochs, self.lossTrainAverage, self.lossValAverage))
            self.valData = self.valData.shuffle(10000)

            self._save()


if __name__ == '__main__':
    def str2bool(v):
        return v.lower() in ("yes", "true", "t", "1")

    parser = argparse.ArgumentParser()
    parser.add_argument('--numEpochs', default=100, type=int)
    parser.add_argument('--lr', default=1e-4, type=float)
    parser.add_argument('--rawDir', default='../data/raw/kaggle', type=str)
    parser.add_argument('--proDir', default='../data/processed/kaggle', type=str)
    parser.add_argument('--batchSize', default=32, type=int)
    parser.add_argument('--preFetch', default=1, type=int)
    parser.add_argument('--valSize', default=1000, type=int)
    parser.add_argument('--threadNum', default=4, type=int)
    parser.add_argument('--restore', default=True, type=str2bool)
    args = parser.parse_args()

    train = TrainPose(args)

    try:
        train.train()
    except KeyboardInterrupt:
        print('')
        print("Training stopped. Continue by running again with --restore True. Note: will reset weight decay for AdamW optimizer")
