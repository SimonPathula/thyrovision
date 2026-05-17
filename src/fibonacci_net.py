import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.layers import Conv2D, ReLU, MaxPooling2D, Concatenate, Dense, BatchNormalization, AveragePooling2D, GlobalAveragePooling2D, Dropout

def fibonacci_filters(n):
    fib = [21, 34]
    while len(fib) < n:
        fib.append(fib[-1] + fib[-2])
    return fib  # [21, 34, 55, 89, 144, 233, 377] for n=7
 
 

class FibonacciConvBlock(layers.Layer):
 
    def __init__(self, filters):
        super().__init__()
 
        self.conv = Conv2D(
            filters=filters,
            kernel_size=(3, 3),
            padding="same",
            kernel_initializer="he_normal"
        )
        self.bn = BatchNormalization()
        self.relu = ReLU()
        self.pool = MaxPooling2D((2, 2))
 
    def call(self, inputs, training=False):
        x = self.conv(inputs)
        x = self.bn(x, training=training)
        x = self.relu(x)
        x = self.pool(x)
        return x
 
class Avg2MaxPooling(layers.Layer):
 
    def __init__(self):
        super().__init__()
 
        self.avg_pool = AveragePooling2D(pool_size=3, strides=2, padding="same")
        self.max_pool = MaxPooling2D(pool_size=3, strides=2, padding="same")
 
    def call(self, inputs):
        avg = self.avg_pool(inputs)
        mx  = self.max_pool(inputs)
        return avg - 2 * mx
 
class PCB1(layers.Layer):
 
    def __init__(self):
        super().__init__()
        self.avg2max   = Avg2MaxPooling()
        self.concat = Concatenate(axis=-1)
 
    def call(self, skip_input, main_input):
        skip = self.avg2max(skip_input)
        return self.concat([main_input, skip])

class PCB2(layers.Layer):
 
    def __init__(self, branch_filters=24):
        super().__init__()
 
        self.conv    = Conv2D(branch_filters, kernel_size=3, padding='same',
                              kernel_initializer='he_normal')
        self.bn      = BatchNormalization()
        self.relu    = ReLU()
        self.avg2max = Avg2MaxPooling()
        self.concat  = Concatenate(axis=-1)
 
    def call(self, skip_input, main_input, training=False):

        x = self.conv(skip_input)
        x = self.relu(x)
        x = self.bn(x, training=training)
        x = self.avg2max(x)
        return self.concat([main_input, x])
 
class DWSCBlock(layers.Layer):
 
    def __init__(self, filters, depth_multiplier):
        super().__init__()

        self.depthwise = layers.DepthwiseConv2D(
            depth_multiplier = depth_multiplier,
            kernel_size=(3, 3),
            padding='same'
        )

        self.pointwise = Conv2D(
            filters=filters,
            kernel_size=(1, 1),
            padding='same'
        )

        self.bn = BatchNormalization()
        self.relu = ReLU()

    def call(self, inputs, training=False):
        x = self.depthwise(inputs)
        x = self.pointwise(x)
        x = self.bn(x, training=training)
        x = self.relu(x)
        return x

class FibonacciNet(Model):
 
    def __init__(self, num_classes=1, activation = "sigmoid", **kwargs):
        super().__init__(**kwargs)
 
        fib = fibonacci_filters(7)  # [21, 34, 55, 89, 144, 233, 377]
 
        self.block1 = FibonacciConvBlock(fib[0])
        self.block2 = FibonacciConvBlock(fib[1])
        self.pcb1   = PCB1()
        self.block3 = FibonacciConvBlock(fib[2])
        self.pcb2   = PCB2(branch_filters=24)
        self.block4 = FibonacciConvBlock(fib[3])
        self.block5 = FibonacciConvBlock(fib[4])
 
        self.dwsc1  = DWSCBlock(fib[5], 1)
        self.dwsc2  = DWSCBlock(fib[6], 2)          
 
        self.gap        = GlobalAveragePooling2D()
        self.dropout    = Dropout(0.5)
        self.classifier = Dense(num_classes, activation)
 
    def call(self, inputs, training=False):
 
        x1 = self.block1(inputs, training=training)
 
        x2 = self.block2(x1, training=training)

        x3 = self.block3(x2, training=training)
 
        x = self.pcb1(x2, x3)
 
        x = self.block4(x, training=training)
 
        x = self.pcb2(x3, x, training=training)

        x = self.block5(x, training= training)
 
        x = self.dwsc1(x, training=training)
        x = self.dwsc2(x, training=training)
 
        x = self.gap(x)
        x = self.dropout(x, training=training)
        x = self.classifier(x)
 
        return x