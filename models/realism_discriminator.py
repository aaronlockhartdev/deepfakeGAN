import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import (Conv2D)


class RealismDiscriminator(Model):
    def __init__(self):
        super().__init__()
        self.conv1 = Conv2D(128, (5, 5))

    def call(self):
        return
