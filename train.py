import tensorflow as tf
import numpy as np
from time import time
from model import create_generator, create_discriminator
from get_data import load_data
from constants import *
from progress.bar import IncrementalBar



def train():
    gpu = tf.config.experimental.list_physical_devices('GPU')[0]
    tf.config.experimental.set_memory_growth(gpu, True)
    generator = create_generator()
    discriminator = create_discriminator()

    cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)

    def discriminator_loss(realOut, fakeOut):
        realLoss = cross_entropy(tf.ones_like(realOut), realOut)
        fakeLoss = cross_entropy(tf.zeros_like(fakeOut), fakeOut)
        totalLoss = realLoss + fakeLoss

        return totalLoss

    def generator_loss(fakeOut):

        return cross_entropy(tf.ones_like(fakeOut), fakeOut)


    generatorOptimizer = tf.keras.optimizers.Adam(1e-4)
    discriminatorOptimizer = tf.keras.optimizers.Adam(1e-4)

    checkpoint = tf.train.Checkpoint(generator_optimizer=generatorOptimizer,
                                     discriminator_optimizer=discriminatorOptimizer,
                                     generator=generator,
                                     discriminator=discriminator)

    data = load_data()

    print("\nData Loaded")

    np.random.shuffle(data)

    print("\nData Shuffled")

    print(data.shape[0])
    data = data[:data.shape[0] - (data.shape[0] % BATCH_SIZE)]
    print(data.shape[0])
    numSplits = int(data.shape[0] / BATCH_SIZE)
    print(numSplits)

    splitData = tf.split(data, num_or_size_splits=numSplits, axis=0)


    @tf.function
    def train_batch(data):

        with tf.GradientTape() as genTape, tf.GradientTape() as discTape:
            generated = generator([data, data], training=True)

            realOut = discriminator(data, training=True)
            fakeOut = discriminator(generated, training=True)

            genLoss = generator_loss(fakeOut)
            discLoss = discriminator_loss(realOut, fakeOut)

        generatorGradients = genTape.gradient(genLoss, generator.trainable_variables)
        discriminatorGradients = discTape.gradient(discLoss, discriminator.trainable_variables)

        generatorOptimizer.apply_gradients(zip(generatorGradients, generator.trainable_variables))
        discriminatorOptimizer.apply_gradients(zip(discriminatorGradients, discriminator.trainable_variables))

        return genLoss, discLoss

    for epoch in range(EPOCHS):
        start = time()
        bar = IncrementalBar("Training epoch " + str(epoch), max=numSplits)
        genLossSum = 0
        discLossSum = 0
        lossCount = 0
        for batch in splitData:
            genLoss, discLoss = train_batch(batch)
            genLossSum += genLoss
            discLossSum += discLoss
            lossCount += 1
            bar.next()
        end = time()

        print("\nEpoch " + str(epoch) + " took " + str(end - start) + " seconds with a generator loss of " + str(genLossSum/lossCount) + " and a discriminator loss of " + str(discLossSum/lossCount))
