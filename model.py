import tensorflow as tf
from tensorflow.keras import layers
from constants import *

def create_generator():

    #print("SOURCE MODEL")

    source_input = layers.Input(shape = (IMAGE_WIDTH, IMAGE_HEIGHT, 3))
    source = layers.BatchNormalization()(source_input)

    source = layers.Conv2D(256, (5, 5), strides=(1, 1), padding='SAME', use_bias=True)(source)
    source = layers.LeakyReLU()(source)
    source = layers.BatchNormalization()(source)
    source = layers.MaxPooling2D(pool_size=(2, 2), strides=None, padding='SAME')(source)

    #print(source.shape)

    source = layers.Conv2D(256, (3, 3), strides=(1, 1), padding='SAME', use_bias=True)(source)
    source = layers.LeakyReLU()(source)
    source = layers.BatchNormalization()(source)
    source = layers.MaxPooling2D(pool_size=(2, 2), strides=None, padding='SAME')(source)

    #print(source.shape)

    source = layers.Conv2D(256, (2, 2), strides=(1, 1), padding='SAME', use_bias=True)(source)
    source = layers.LeakyReLU()(source)
    source = layers.BatchNormalization()(source)
    source = layers.MaxPooling2D(pool_size=(2, 2), strides=None, padding='SAME')(source)

    #print(source.shape)

    source = layers.Flatten()(source)

    source = layers.Dense(128, activation='sigmoid')(source)

    #print(source.shape)

    source = layers.Dense(72, activation='sigmoid')(source)

    #print(source.shape)

    #print("TARGET MODEL")

    target_input = layers.Input(shape = (IMAGE_WIDTH, IMAGE_HEIGHT, 3))
    target = layers.BatchNormalization()(target_input)

    target = layers.Conv2D(256, (5, 5), strides=(1, 1), padding='SAME', use_bias=True)(target)
    target = layers.LeakyReLU()(target)
    target = layers.BatchNormalization()(target)
    target = layers.MaxPooling2D(pool_size=(2, 2), strides=None, padding='SAME')(target)

    #print(target.shape)

    target = layers.Conv2D(256, (3, 3), strides=(1, 1), padding='SAME', use_bias=True)(target)
    target = layers.LeakyReLU()(target)
    target = layers.BatchNormalization()(target)
    target = layers.MaxPooling2D(pool_size=(2, 2), strides=None, padding='SAME')(target)

    #print(target.shape)

    target = layers.Conv2D(256, (2, 2), strides=(1, 1), padding='SAME', use_bias=True)(target)
    target = layers.LeakyReLU()(target)
    target = layers.BatchNormalization()(target)
    target = layers.MaxPooling2D(pool_size=(2, 2), strides=None, padding='SAME')(target)

    #print(target.shape)

    target = layers.Flatten()(target)

    target = layers.Dense(128, activation='sigmoid')(target)

    #print(target.shape)

    target = layers.Dense(72, activation='sigmoid')(target)

    #print(target.shape)

    #print("MERGED MODEL")

    merged = layers.concatenate([source, target])

    #print(merged.shape)

    merged = layers.Dense(72, activation='sigmoid')(merged)

    #print(merged.shape)

    merged = layers.Dense(86, activation='sigmoid')(merged)

    #print(merged.shape)

    merged = layers.Dense(100, activation='sigmoid')(merged)

    #print(merged.shape)

    merged = layers.Reshape((10, 10, 256))(merged)

    merged = layers.Conv2DTranspose(256, (5, 5), strides=(2, 2), padding='same', use_bias=True)(merged)

    #print(merged.shape)

    merged = layers.Conv2DTranspose(64, (5, 5), strides=(2, 2), padding='same', use_bias=True)(merged)

    #print(merged.shape)

    merged = layers.Conv2DTranspose(3, (5, 5), strides=(2, 2), padding='same', use_bias=True)(merged)

    #print(merged.shape)

    model = tf.keras.Model(inputs=[source_input, target_input], outputs=merged)

    print(model.summary())

    model.compile(optimizer='adam', loss='mse')

    return model

def create_descriminator():
    return
