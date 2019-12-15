import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import (BatchNormalization, Conv2D, Activation,
                                     MaxPooling2D, Dropout, Flatten, Dense)


class FeaturePose(Model):
    def __init__(self):
        super().__init__()
        self.bn1 = BatchNormalization(input_shape=(96, 96, 1))
        self.conv1 = Conv2D(24, (5, 5), kernel_initializer='he_normal')
        self.a1 = Activation('relu')
        self.mp1 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))
        self.drop1 = Dropout(0.2)

        self.conv2 = Conv2D(36, (5, 5))
        self.a2 = Activation('relu')
        self.mp2 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))
        self.drop2 = Dropout(0.2)

        self.conv3 = Conv2D(48, (5, 5))
        self.a3 = Activation('relu')
        self.mp3 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))
        self.drop3 = Dropout(0.2)

        self.conv4 = Conv2D(64, (3, 3))
        self.a4 = Activation('relu')
        self.mp4 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))
        self.drop4 = Dropout(0.2)

        self.conv5 = Conv2D(64, (3, 3))
        self.a5 = Activation('relu')

        self.f1 = Flatten()

        self.dense1 = Dense(500, activation='relu')
        self.dense2 = Dense(90, activation='relu')
        self.dense3 = Dense(30)

    def call(self, input, training=False):

        x = self.bn1(input)
        x = self.conv1(x)
        x = self.a1(x)
        x = self.mp1(x)
        if training:
            x = self.drop1(x)

        x = self.conv2(x)
        x = self.a2(x)
        x = self.mp2(x)
        if training:
            x = self.drop2(x)

        x = self.conv3(x)
        x = self.a3(x)
        x = self.mp3(x)
        if training:
            x = self.drop3(x)

        x = self.conv4(x)
        x = self.a4(x)
        x = self.mp4(x)
        if training:
            x = self.drop4(x)

        x = self.conv5(x)
        x = self.a5(x)

        x = self.f1(x)

        x = self.dense1(x)
        x = self.dense2(x)
        x = self.dense3(x)

        return x
