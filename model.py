import tensorflow as tf
from tensorflow.keras import layers
from constants import *

def create_generator():

    source_input = layers.Input(shape=(IMAGE_WIDTH, IMAGE_HEIGHT, 3))
    source = layers.BatchNormalization()(source_input)

    source = layers.Conv2D(64, (5, 5), strides=(1, 1), padding='SAME', use_bias=True)(source)
    source = layers.LeakyReLU()(source)
    source = layers.BatchNormalization()(source)
    source = layers.MaxPooling2D(pool_size=(2, 2), strides=None, padding='SAME')(source)

    source = layers.Conv2D(256, (3, 3), strides=(1, 1), padding='SAME', use_bias=True)(source)
    source = layers.LeakyReLU()(source)
    source = layers.BatchNormalization()(source)
    source = layers.MaxPooling2D(pool_size=(2, 2), strides=None, padding='SAME')(source)

    source = layers.Conv2D(128, (2, 2), strides=(1, 1), padding='SAME', use_bias=True)(source)
    source = layers.LeakyReLU()(source)
    source = layers.BatchNormalization()(source)
    source = layers.MaxPooling2D(pool_size=(2, 2), strides=None, padding='SAME')(source)

    source = layers.Flatten()(source)

    source = layers.Dense(800, activation='sigmoid')(source)

    source = layers.Dense(600, activation='sigmoid')(source)

    target_input = layers.Input(shape=(IMAGE_WIDTH, IMAGE_HEIGHT, 3))
    target = layers.BatchNormalization()(target_input)

    target = layers.Conv2D(64, (11, 11), strides=(1, 1), padding='SAME', use_bias=True)(target)
    target = layers.LeakyReLU()(target)
    target = layers.BatchNormalization()(target)
    target = layers.MaxPooling2D(pool_size=(2, 2), strides=None, padding='SAME')(target)

    target = layers.Conv2D(256, (9, 9), strides=(1, 1), padding='SAME', use_bias=True)(target)
    target = layers.LeakyReLU()(target)
    target = layers.BatchNormalization()(target)
    target = layers.MaxPooling2D(pool_size=(2, 2), strides=None, padding='SAME')(target)

    target = layers.Conv2D(128, (7, 7), strides=(1, 1), padding='SAME', use_bias=True)(target)
    target = layers.LeakyReLU()(target)
    target = layers.BatchNormalization()(target)
    target = layers.MaxPooling2D(pool_size=(2, 2), strides=None, padding='SAME')(target)

    target = layers.Flatten()(target)

    target = layers.Dense(800, activation='sigmoid')(target)

    target = layers.Dense(600, activation='sigmoid')(target)

    merged = layers.concatenate([source, target])

    merged = layers.Dense(800, activation='sigmoid')(merged)

    merged = layers.Dense(600, activation='sigmoid')(merged)

    merged = layers.Dense(400, activation='sigmoid')(merged)

    merged = layers.Reshape((20, 20, 1))(merged)

    merged = layers.Conv2DTranspose(64, (5, 5), strides=(2, 2), padding='same', use_bias=True)(merged)

    merged = layers.Conv2DTranspose(256, (5, 5), strides=(3, 3), padding='same', use_bias=True)(merged)

    merged = layers.Conv2DTranspose(128, (5, 5), strides=(2, 2), padding='same', use_bias=True)(merged)

    merged = layers.Conv2DTranspose(3, (3, 3), strides=(1, 1), padding='same', use_bias=True)(merged)

    model = tf.keras.Model(inputs=[source_input, target_input], outputs=merged)

    print(model.summary())

    return model

def create_discriminator():

    input = layers.Input(shape=(IMAGE_WIDTH, IMAGE_HEIGHT, 3))

    discriminator = layers.Conv2D(64, (5, 5), strides=(2, 2), padding='same')(input)

    discriminator = layers.LeakyReLU()(discriminator)

    discriminator = layers.Dropout(0.3)(discriminator)

    discriminator = layers.Conv2D(64, (5, 5), strides=(2, 2), padding='same')(discriminator)

    discriminator = layers.LeakyReLU()(discriminator)

    discriminator = layers.Dropout(0.3)(discriminator)

    discriminator = layers.Flatten()(discriminator)

    discriminator = layers.Dense(1)(discriminator)

    model = tf.keras.Model(inputs=[input], outputs=discriminator)

    print(model.summary())
