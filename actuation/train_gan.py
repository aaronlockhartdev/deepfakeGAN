import os
import csv
import argparse
import tensorflow as tf
from tf import GradientTape
from tf.keras.optimizers import Adam
from tf.keras.losses import BinaryCrossentropy
from tf.keras.utils import normalize
from tf.image import (rgb_to_grayscale, resize)


class TrainGAN():

    def __init__(self, args):
        # import classes
        sys.path.append("..")
        from preprocessing.process_vids import ProcessVids
        from models.generator import Generator
        from models.realism_discriminator import RealismDiscriminator
        from models.feature_pose import FeaturePose
        # initialize training variables
        self.numEpochs = args.numEpochs

        self.genOptimizer = Adam(args.lr)
        self.realismOptimizer = Adam(args.lr)

        self.genLoss = self.realismLoss = None

        self.crossEntropy = BinaryCrossentropy(from_logits=True)

        # initialize preprocessor and build/load data
        preprocessor = ProcessVids(args.rawDir, args.proDir,
                                   args.batchSize, args.valSize,
                                   args.threadNum, args.preFetch)
        self.trainData, self.valData = preprocessor.loadData()

        # initialize models
        self.generator = Generator()
        self.realism = RealismDiscriminator()
        self.featurePose = FeaturePose()

        # initialize checkpoints and managers
        if not args.restore:
            for f in os.listdir('data/checkpoints/generator'):
                os.remove('data/checkpoints/generator/' + f)
            for f in os.listdir('data/checkpoints/realism_discriminator'):
                os.remove('data/checkpoints/realism_discriminator/' + f)
            for f in os.listdir('data/checkpoints/feature_pose'):
                os.remove('data/checkpoints/feature_pose/' + f)

        self.checkpointG = tf.train.Checkpoint(optimizer=self.genOptimizer,
                                               model=self.generator)
        self.managerG = tf.train.CheckpointManager(self.checkpointG,
                                                   directory='data/checkpoints/generator',
                                                   max_to_keep=5, keep_checkpoint_every_n_hours=2)

        self.checkpointR = tf.train.Checkpoint(optimizer=self.realismOptimizer,
                                               model=self.realism)
        self.managerR = tf.train.CheckpointManager(self.checkpointR,
                                                   directory='data/checkpoints/realism_discriminator',
                                                   max_to_keep=5, keep_checkpoint_every_n_hours=2)

        if os.path.isfile('/data/checkpoints/gan_losses.csv'):
            os.remove('/data/checkpoints/gan_losses.csv')

    def _generatorLoss(self, x):
        return self.crossEntropy(tf.ones_like(x), x)

    def _realismLoss(self, realOut, fakeOut):
        realLoss = self.crossEntropy(tf.ones_like(realOut), realOut)
        fakeOut = self.crossEntropy(tf.zeros_like(fakeOut), fakeOut)
        return realLoss + fakeLoss

    def _compareFeatures(self, x, y):
        x = normalize(x)
        y = normalize(y)
        diff = tf.math.subtract(x, y)
        diff = tf.math.abs(diff)
        diff = tf.math.reduce_mean(diff)
        return self.crossEntropy(tf.zeros_like(diff), diff)

    def _logTrain(self, epoch):
        msg = 'Training Epoch {}, Generator Loss: {}, Discriminator Loss: {}'
        print(msg.format(epoch + 1, self.genLoss, self.realismLoss), end='\r')

    def _logVal(self, epoch):
        msg = 'Validating Epoch {}, Generator Loss: {}, Discriminator Loss: {}'
        print(msg.format(epoch + 1, self.genLoss, self.realismLoss), end='\r')

    def _save(self):
        self.managerG.save()
        self.managerR.save()
        with open('data/checkpoints/gan_losses.csv', 'a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([self.genLoss, self.realismLoss])

    def _greyscale(self, image):
        image = resize(image, (96, 96))
        image = rgb_to_grayscale(image)
        return image

    @tf.function
    def _validate(self, inputA, inputB):
        generated = self.generator.call(inputA, inputB, training=True)

        realismReal = self.realism.call(input, training=True)
        realismFake = self.realism.call(generated, training=True)

        genLossR = self._generatorLoss(realismFake)

        self.realismLoss = self._realismLoss(realismReal, realismFake)

        featuresSource = self.featurePose.call(inputB)
        featuresGen = self.featurePose.call(generated)

        featuresfeaturePose = _compareFeatures(featuresSource, featuresGen)

        genLossS = self._generatorLoss(featuresfeaturePose)

        self.genLoss = (genLossR + genLossS) / 2

    @tf.function
    def _update(self, inputA, inputB):
        with GradientTape() as genTape, GradientTape() as realismTape:
            generated = self.generator.call(inputA, inputB, training=True)

            realismReal = self.realism.call(inputA, training=True)
            realismFake = self.realism.call(generated, training=True)

            genLossR = self._generatorLoss(realismFake)

            self.realismLoss = self._realismLoss(realismReal, realismFake)

            featuresSource = self.featurePose.call(self._greyscale(inputB))
            featuresGen = self.featurePose.call(self._greyscale(generated))

            featurePose = _compareFeatures(featuresSource, featuresGen)

            genLossS = self._generatorLoss(featurePose)

            self.genLoss = (genLossR + genLossS) / 2

            genGradients = genTape.gradient(self.genLoss, self.generator.trainable_variables)
            realismGradients = realismTape.gradient(
                self.realismLoss, self.realism.trainable_variables)

            self.genOptimizer.apply_gradients(zip(genGradients, self.generator.trainable_variables))
            self.realismOptimizer.apply_gradients(
                zip(realismGradients, self.realism.trainable_variables))

    def train(self):
        self.checkpointG.restore(self.managerG.latest_checkpoint)
        self.checkpointR.restore(self.managerR.latest_checkpoint)

        for epoch in range(self.numEpochs):
            for input, output in self.trainData:
                self._update(input, output)
                self._logTrain(epoch)
            print('')
            for input, output in self.valData:
                self._validate(input, output)
                self._logVal(epoch)

            print('Epoch {} complete'.format(epoch + 1))
            self.trainData = self.trainData.shuffle(args.threadNum)
            self.valData = self.valData.shuffle(args.threadNum)

            self._save()


if __name__ == '__main__':
    def str2bool(v):
        return v.lower() in ("yes", "true", "t", "1")

    parser = argparse.ArgumentParser()
    parser.add_argument('--numEpochs', default=100, type=int)
    parser.add_argument('--lr', default=1e-4, type=float)
    parser.add_argument('--rawDir', default='data/raw/faces', type=str)
    parser.add_argument('--proDir', default='data/processed/faces', type=str)
    parser.add_argument('--batchSize', default=32, type=int)
    parser.add_argument('--preFetch', default=1, type=int)
    parser.add_argument('--valSize', default=1000, type=int)
    parser.add_argument('--threadNum', default=4, type=int)
    parser.add_argument('--restore', default=True, type=str2bool)
    args = parser.parse_args()

    train = TrainGAN(args)
    train.train()
