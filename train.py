import tensorflow as tf
import numpy as np
from time import time
from model import create_generator, create_discriminator
from get_data import load_data
from constants import *


def train():
    generator = create_generator()
    discriminator = create_discriminator()

    cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)

    generatorOptimizer = tf.keras.optimizers.Adam(1e-4)
    discriminatorOptimizer = tf.keras.optimizers.Adam(1e-4)

    checkpoint = tf.train.Checkpoint(generator_optimizer=generatorOptimizer,
                                     discriminator_optimizer=discriminatorOptimizer,
                                     generator=generator,
                                     discriminator=discriminator)

    data = load_data()

    np.random.shuffle(data)

    numSplits = int(data.shape[0] / BATCH_SIZE)

    splitData = tf.split(data, num_or_size_splits=numSplits, axis=1)

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

    for epoch in range(EPOCHS):
        start = time()
        for batch in splitData:
            train_batch(batch)
        end = time()

        print("Epoch " + str(epoch) + " took " + str(end - start) + " seconds")





def discriminator_loss(realOut, fakeOut):
    global cross_entropy

    realLoss = cross_entropy(tf.ones_like(realOut), realOut)
    fakeLoss = cross_entropy(tf.zeros_like(fakeOut), fakeOut)
    totalLoss = realLoss + fakeLoss

    return totalLoss

def generator_loss(fakeOut):
    global cross_entropy

    return cross_entropy(tf.ones_like(fakeOut), fakeOut)
