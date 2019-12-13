import os
import csv
import sys
import argparse
import tensorflow as tf
from tensorflow import GradientTape
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import MSE


class TrainPose():
    def __init__(self, args):
        # import classes
        sys.path.append("..")
        from models.feature_pose import FeaturePose
        from preprocessing.process_kaggle import ProcessKaggle
        # initialize training variables
        self.numEpochs = args.numEpochs

        self.optimizer = Adam(args.lr)

        self.loss = None

        self.mse = MSE

        # initialize preprocessor and build/load data
        preprocessor = ProcessKaggle(args.rawDir, args.proDir,
                                     args.batchSize, args.threadNum,
                                     args.preFetch, args.valSize)

        self.trainData, self.valData = preprocessor.loadData()

        # initialize models

        self.model = FeaturePose()

        # initialize checkpoints and managers
        if not args.restore:
            for f in os.listdir('data/checkpoints/feature_pose'):
                os.remove('data/checkpoints/feature_pose/' + f)

        self.checkpoint = tf.train.Checkpoint(optimizer=self.optimizer,
                                              model=self.optimizer)
        self.manager = tf.train.CheckpointManager(self.checkpoint,
                                                  directory='data/checkpoints/feature_pose',
                                                  max_to_keep=5, keep_checkpoint_every_n_hours=2)

        if os.path.isfile('/data/checkpoints/feature_pose_losses.csv'):
            os.remove('/data/checkpoints/feature_pose_losses.csv')

    def _loss(self, x, y):
        return self.mse(y, x)

    def _logTrain(self, epoch):
        msg = 'Training Epoch {}, Model Loss: {}'
        print(msg.format(epoch + 1, self.loss), end='\r')

    def _logTrain(self, epoch):
        msg = 'Validating Epoch {}, Model Loss: {}'
        print(msg.format(epoch + 1, self.loss), end='\r')

    def _save(self):
        self.manager.save()
        with open('data/checkpoints/feature_pose_losses.csv', 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([self.loss])

    @tf.function
    def _validate(self, input, output):
        predicted = self.model.call(input)
        self.loss = _loss(predicted, output)

    @tf.function
    def _update(self, input, ouput):
        with GradientTape() as tape:
            predicted = self.model.call(input)
            self.loss = _loss(predicted, output)
            gradients = tape.gradient(self.loss, self.model.trainable_variables)
            self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))

    def train(self):
        self.checkpoint.restore(self.manager.latest_checkpoint)

        for epoch in range(self.numEpochs):
            for input, output in self.trainData:
                self._update(input, output)
                self._logTrain(epoch)
            print('')
            for input, output in self.valData:
                self._validate(input, output)
                self._logVal(epoch)

            print('Epoch {} complete'.format(epoch + 1))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--numEpochs', default=100, type=int)
    parser.add_argument('--lr', default=1e-4, type=float)
    parser.add_argument('--rawDir', default='../data/raw/kaggle', type=str)
    parser.add_argument('--proDir', default='../data/processed/kaggle', type=str)
    parser.add_argument('--batchSize', default=32, type=int)
    parser.add_argument('--preFetch', default=1, type=int)
    parser.add_argument('--valSize', default=1000, type=int)
    parser.add_argument('--threadNum', default=4, type=int)
    parser.add_argument('--restore', default=True, type=bool)
    args = parser.parse_args()

    train = TrainPose(args)
    train.train()
