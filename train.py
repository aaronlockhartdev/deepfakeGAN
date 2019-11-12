import tensorflow as tf
import numpy as np
from time import time
from model import create_generator, create_discriminator
from get_data import load_batches, load_data
from constants import *
from progress.bar import IncrementalBar
import dask.array as da



def train():
    tf.executing_eagerly()
    tf.keras.backend.set_floatx('float16')
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

    if BATCH_SIZE == 1:
        batches = load_data()
    else:
        batches = load_batches()


    print(batches[0].shape)

    print("\nData Loaded")


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

    def train_dataset(batches):
        for epoch in range(EPOCHS):
            randomized = np.arange(batches.shape[0])
            np.random.shuffle(randomized)

            start = time()
            genLossSum = 0
            discLossSum = 0
            lossCount = 0

            bar = IncrementalBar("Training epoch " + str(epoch), max=batches.shape[0])


            for i in range(batches.shape[0]):
                batch = tf.constant(batches[randomized[i]], shape=(BATCH_SIZE, IMAGE_WIDTH, IMAGE_HEIGHT, 3))
                genLoss, discLoss = train_batch(batch)
                genLossSum += genLoss
                discLossSum += discLoss
                lossCount += 1
                bar.next()
            if (epoch + 1) % 15 == 0:
                checkpoint.save(file_prefix = DATA_OUTPUT_PATH + "/checkpoints/")
            print('\n\nTime for epoch {} is {} sec'.format(epoch + 1, time()-start))
            print('Loss is {} (generator) and {} (discriminator)\n'.format(genLossSum/lossCount, discLossSum/lossCount))

    train_dataset(batches)
