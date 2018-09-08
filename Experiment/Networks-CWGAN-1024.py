import random as rd
import tensorflow as tf


BATCH_SIZE = -1
CHANNELS = 1
CLASSES = 2
KERNEL_SIZE = 25
MODEL_SIZE = 16
PHASE_SHUFFLE = 2
STRIDE = 4
WAV_LENGTH = 1024
Z_LENGTH = 100


def generator(x, y):
    """ A waveGAN generator """

    x = tf.cast(x, tf.float32)
    y = tf.cast(y, tf.float32)

    # Input: [64, 100] > [64, 992]
    # densify = tf.layers.dense(
    #    inputs=x,
    #    units=WAV_LENGTH - (CLASSES * MODEL_SIZE),
    #    name="Z-Input"
    # )

    densify = tf.layers.dense(
        inputs=x,
        units=WAV_LENGTH - CLASSES,
        name="Z-Input"
    )

    # INPUT: [64, 992] > [64, 1, 16, 62]
    # shape = tf.reshape(
    #     tensor=densify,
    #    shape=[BATCH_SIZE, 1, 16, (MODEL_SIZE * 4) - CLASSES]
    # )
    shape = tf.reshape(
        tensor=densify,
        shape=[BATCH_SIZE, 1, 1, WAV_LENGTH - CLASSES]
    )

    y = y[:, :, 0:1, :]

    # Input: [64, 1, 1, 992] > [64, 1, 1, 1024]
    concat = tf.concat(values=[shape, y], axis=3)

    layer = tf.nn.relu(concat)

    # Input: [64, 1, 1, 1024] > [64, 1, 16, 64]
    layer = tf.layers.conv2d_transpose(
        inputs=layer,
        filters=MODEL_SIZE * 4,
        kernel_size=(1, 16),  # WHAT SHOULD THIS BE???
        strides=(1, 1),
        padding='valid',
        name="TransConvolution"
    )

    print(layer)

    # Input: [64, 1, 16, 64] > [64, 1, 64, 32]
    layer = tf.layers.conv2d_transpose(
        inputs=layer,
        filters=MODEL_SIZE * 2,
        kernel_size=(1, KERNEL_SIZE),
        strides=(1, STRIDE),
        padding='same',
        name="TransConvolution1"
    )

    layer = tf.nn.relu(layer)

    # Input: [64, 1, 64, 32] > [64, 1, 256, 16]
    layer = tf.layers.conv2d_transpose(
        inputs=layer,
        filters=MODEL_SIZE,
        kernel_size=(1, KERNEL_SIZE),
        strides=(1, STRIDE),
        padding='same',
        name="TransConvolution2"
    )

    layer = tf.nn.relu(layer)

    # Input: [64, 1, 256, 16] > [64, 1024, 1]
    layer = tf.layers.conv2d_transpose(
        inputs=layer,
        filters=CHANNELS,
        kernel_size=(1, KERNEL_SIZE),
        strides=(1, STRIDE),
        padding='same',
        name="TransConvolution3"
    )[:, 0]

    # Input: [64, 1024, 1]
    tanh = tf.tanh(
        x=layer,
        name="GeneratedSamples"
    )

    return tanh


def discriminator(x, y):
    """ A waveGAN discriminator """

    # x = x + tf.random_normal(
    #     shape=tf.shape(x),
    #     mean=0.0,
    #     stddev=0.2,
    #     dtype=tf.float32
    # )

    x = tf.print(x, [x])
    y = tf.print(y, [y])
    layer = tf.multiply(x, y)
    layer = tf.print(layer, [layer])

    # layer = tf.concat(values=[x, y], axis=2)

    # Input: [64, 1024, 3] > [64, 256, 16]
    layer = tf.layers.conv1d(
        inputs=layer,
        filters=MODEL_SIZE,
        kernel_size=KERNEL_SIZE,
        strides=STRIDE,
        padding='same'
    )
    layer = _leakyRelu(layer)
    layer = _phaseShuffle(layer)

    # Input: [64, 256, 16] > [64, 64, 32]
    layer = tf.layers.conv1d(
        inputs=layer,
        filters=MODEL_SIZE * 2,
        kernel_size=KERNEL_SIZE,
        strides=STRIDE,
        padding='same'
    )
    layer = _leakyRelu(layer)
    layer = _phaseShuffle(layer)

    # Input: [64, 64, 32] > [64, 16, 64]
    layer = tf.layers.conv1d(
        inputs=layer,
        filters=MODEL_SIZE * 4,
        kernel_size=KERNEL_SIZE,
        strides=STRIDE,
        padding='same'
    )
    layer = _leakyRelu(layer)

    # Input: [64, 16, 64] > [64, 1, 1]
    layer = tf.layers.conv1d(
        inputs=layer,
        filters=1,
        kernel_size=16,
        strides=1,
        padding='valid'
    )[:, 0]
    # print('new layer ' + str(layer))

    # Input: [64, 16, 64] > [64, 1024]
    # layer = tf.reshape(
    #     tensor=layer,
    #     shape=[BATCH_SIZE, WAV_LENGTH]
    # )

    # Input: [64, 1024] > [64, 1]
    logits = tf.layers.dense(
        inputs=layer,
        units=1
    )[:, 0]

    # logits = tf.nn.sigmoid(layer)
    # print('logits ' + str(logits))

    # logits = tf.print(
    #     logits,
    #     [logits]
    # )

    return logits


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
