import random as rd
import tensorflow as tf


BATCH_SIZE = -1
CHANNELS = 1
CLASSES = 2
KERNEL_SIZE = 25
MODEL_SIZE = 32
PHASE_SHUFFLE = 2
STRIDE = 4
WAV_LENGTH = 4096
Z_LENGTH = 100


def generator(x, y):
    """ A waveGAN generator """

    x = tf.cast(x, tf.float32)
    y = tf.cast(y, tf.float32)

    # Input: [64, 100] > [64, 4094]
    densify = tf.layers.dense(
        inputs=x,
        units=WAV_LENGTH - CLASSES,
        name="Z-Input"
    )

    # Input: [64, 4032] > [64, 16, 254]
    shape = tf.reshape(
        tensor=densify,
        shape=[BATCH_SIZE, 1, 1, WAV_LENGTH - CLASSES]
    )

    y = y[:, :, 0:1, :]

    # Input: [64, 1, 1, 4094] > [64, 1, 1, 4096]
    concat = tf.concat(values=[shape, y], axis=3)

    layer = tf.nn.relu(concat)

    # Input: [64, 1, 1, 4096] > [64, 1, 16, 256]
    layer = tf.layers.conv2d_transpose(
        inputs=layer,
        filters=MODEL_SIZE * 8,
        kernel_size=(1, 16),
        strides=(1, 1),
        padding='valid',
        name="TransConvolution0"
    )

    # Input: [64, 1, 16, 256] > [64, 1, 64, 128]
    layer = tf.layers.conv2d_transpose(
        inputs=layer,
        filters=MODEL_SIZE * 4,
        kernel_size=(1, KERNEL_SIZE),
        strides=(1, STRIDE),
        padding='same',
        name="TransConvolution1"
    )

    layer = tf.nn.relu(layer)

    # Input: [64, 1, 64, 128] > [64, 1, 256, 64]
    layer = tf.layers.conv2d_transpose(
        inputs=layer,
        filters=MODEL_SIZE * 2,
        kernel_size=(1, KERNEL_SIZE),
        strides=(1, STRIDE),
        padding='same',
        name="TransConvolution2"
    )

    layer = tf.nn.relu(layer)

    # Input: [64, 1, 256, 64] > [64, 1024, 32]
    layer = tf.layers.conv2d_transpose(
        inputs=layer,
        filters=MODEL_SIZE,
        kernel_size=(1, KERNEL_SIZE),
        strides=(1, STRIDE),
        padding='same',
        name="TransConvolution3"
    )

    layer = tf.nn.relu(layer)

    # Input: [64, 1, 1024, 32] > [64, 4096, 1]
    layer = tf.layers.conv2d_transpose(
        inputs=layer,
        filters=CHANNELS,
        kernel_size=(1, KERNEL_SIZE),
        strides=(1, STRIDE),
        padding='same',
        name="TransConvolution4"
    )[:, 0]

    # Input: [64, 4096, 1]
    tanh = tf.tanh(
        x=layer,
        name="GeneratedSamples"
    )

    return tanh


def discriminator(x, y):
    """ A waveGAN discriminator """

    # x = tf.concat(values=[x, y], axis=2)

    # x = x + tf.random_normal(
    #     shape=tf.shape(x),
    #     mean=0.0,
    #     stddev=0.1
    # )

    # Input: [64, 4096, 1] > [64, 1024, 32]
    layer = tf.layers.conv1d(
        inputs=x,
        filters=MODEL_SIZE,
        kernel_size=KERNEL_SIZE,
        strides=STRIDE,
        padding='same'
    )
    layer = _leakyRelu(layer)
    layer = _phaseShuffle(layer)

    # Input: [64, 1024, 32] > [64, 256, 64]
    layer = tf.layers.conv1d(
        inputs=layer,
        filters=MODEL_SIZE * 2,
        kernel_size=KERNEL_SIZE,
        strides=STRIDE,
        padding='same'
    )
    layer = _leakyRelu(layer)
    layer = _phaseShuffle(layer)

    # Input: [64, 256, 64] > [64, 64, 128]
    layer = tf.layers.conv1d(
        inputs=layer,
        filters=MODEL_SIZE * 4,
        kernel_size=KERNEL_SIZE,
        strides=STRIDE,
        padding='same'
    )
    layer = _leakyRelu(layer)
    layer = _phaseShuffle(layer)

    # Input: [64, 64, 128] > [64, 16, 256]
    layer = tf.layers.conv1d(
        inputs=layer,
        filters=MODEL_SIZE * 8,
        kernel_size=KERNEL_SIZE,
        strides=STRIDE,
        padding='same'
    )

    # Input: [64, 16, 256] > [64, 1, 1]
    disc = tf.layers.conv1d(
        inputs=layer,
        filters=1,
        kernel_size=16,
        strides=1,
        padding='valid'
    )[:, 0]

    # Input: [64, 16, 256] > [64, 4096]
    flatten = tf.reshape(
        tensor=layer,
        shape=[BATCH_SIZE, WAV_LENGTH]
    )

    cat = tf.layers.dense(
        inputs=flatten,
        units=2
    )

    return cat, disc


def _phaseShuffle(layer):
    """ Shuffles the phase of each layer """
    batch, length, channel = layer.get_shape().as_list()
    shuffle = _returnPhaseShuffleValue()
    lft = max(0, shuffle)
    rgt = max(0, -shuffle)
    layer = tf.pad(
        tensor=layer,
        paddings=[[0, 0], [lft, rgt], [0, 0]],
        mode='REFLECT'
    )
    layer = layer[:, rgt:rgt+length]
    layer.set_shape([batch, length, channel])
    return layer


def _leakyRelu(inputs, alpha=0.2):
    """ Creates a leaky relu layer """
    return tf.maximum(inputs * alpha, inputs)


def _returnPhaseShuffleValue():
    """ Returns a a ranom integer in the range decided for phase shuffle"""
    return rd.randint(-PHASE_SHUFFLE, PHASE_SHUFFLE)
