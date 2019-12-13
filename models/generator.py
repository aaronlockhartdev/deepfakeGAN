import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import (
    Input, Dense, Flatten, Conv2D, Concatenate, BatchNormalization, Conv2DTranspose)


class Generator(Model):
    def __init__(self):
        super().__init__()
        # branch A
        self.convA1 = Conv2D(512, (5, 5))
        self.bnA1 = BatchNormalization()

        self.convA2 = Conv2D(256, (4, 4))
        self.bnA2 = BatchNormalization()

        self.convA3 = Conv2D(128, (3, 3))
        self.bnA2 = BatchNormalization()

        self.denseA1 = Dense(200)
        self.denseA2 = Dense(75)

        # branch B
        self.convB1 = Conv2D(512, (5, 5))
        self.bnB1 = BatchNormalization()

        self.convB2 = Conv2D(256, (4, 4))
        self.bnB2 = BatchNormalization()

        self.convB3 = Conv2D(128, (3, 3))
        self.bnB2 = BatchNormalization()

        self.denseB1 = Dense(200)
        self.denseB2 = Dense(75)

        # branch M
        self.denseM1 = Dense(150)
        self.denseM2 = Dense(400)

        self.convM1 = Conv2DTranspose(128, (4, 4))
        self.convM2 = Conv2DTranspose(256, (5, 5))
        self.convM3 = Conv2DTranspose(512, (6, 6))

    def call(self, inputTensorA, inputTensorB, training=False):
        # branch A
        a = self.convA1(inputTensorA)
        a = self.bnA1(a, training=training)

        a = self.convA2(a)
        a = self.bnA2(a, training=training)

        a = self.convA3(a)
        a = self.bnA3(a, training=training)

        # branch B
        b = self.convB1(inputTensorB)
        b = self.bnB1(b, training=training)

        b = self.convB2(b)
        b = self.bnB2(b, training=training)

        b = self.convB3(b)
        b = self.bnA3(b, training=training)

        # branch M
        m = Concatenate((a, b))

        m = self.denseM1(m)
        m = self.denseM2(m)

        m = self.convM1(m)
        m = self.convM2(m)
        m = self.convM3(m)

        return m
