import random as rd
import tensorflow as tf


BATCH_SIZE = -1
CHANNELS = 1
KERNEL_SIZE = 25
MODEL_SIZE = 32
PHASE_SHUFFLE = 2
STRIDE = 4
WAV_LENGTH = 4096
Z_LENGTH = 100


def generator(z):
    """ A waveGAN generator """

    z = tf.cast(z, tf.float32)

    # Input: [64, 100] > [64, 4096]
    densify = tf.layers.dense(
        inputs=z,
        units=WAV_LENGTH,
        name="Z-Input"
    )

    # Input: [64, 4096] > [64, 16, 256]
    shape = tf.reshape(
        tensor=densify,
        shape=[BATCH_SIZE, 16, MODEL_SIZE * 8]
    )

    layer = tf.nn.relu(shape)

    # Input: [64, 16, 256] > [64, 64, 128]
    layer = tf.layers.conv2d_transpose(
        inputs=tf.expand_dims(layer, axis=1),
        filters=MODEL_SIZE * 4,
        kernel_size=(1, KERNEL_SIZE),
        strides=(1, STRIDE),
        padding='SAME',
        name="TransConvolution1"
    )[:, 0]

    layer = tf.nn.relu(layer)

    # Input: [64, 64, 128] > [64, 256, 64]
    layer = tf.layers.conv2d_transpose(
        inputs=tf.expand_dims(layer, axis=1),
        filters=MODEL_SIZE * 2,
        kernel_size=(1, KERNEL_SIZE),
        strides=(1, STRIDE),
        padding='SAME',
        name="TransConvolution2"
    )[:, 0]

    layer = tf.nn.relu(layer)

    # Input: [64, 256, 64] > [64, 1024, 32]
    layer = tf.layers.conv2d_transpose(
        inputs=tf.expand_dims(layer, axis=1),
        filters=MODEL_SIZE,
        kernel_size=(1, KERNEL_SIZE),
        strides=(1, STRIDE),
        padding='SAME',
        name="TransConvolution3"
    )[:, 0]

    layer = tf.nn.relu(layer)

    # Input: [64, 1024, 32] > [64, 4096, 1]
    layer = tf.layers.conv2d_transpose(
        inputs=tf.expand_dims(layer, axis=1),
        filters=CHANNELS,
        kernel_size=(1, KERNEL_SIZE),
        strides=(1, STRIDE),
        padding='SAME',
        name="TransConvolution4"
    )[:, 0]

    # Input: [64, 4096, 1]
    tanh = tf.tanh(
        x=layer,
        name="GeneratedSamples"
    )

    return tanh


def discriminator(features):
    """ A waveGAN discriminator """

    # Input: [64, 4096, 1] > [64, 1024, 32]
    layer = tf.layers.conv1d(
        inputs=features,
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
    layer = _leakyRelu(layer)

    # Input: [64, 16, 256] > [64, 4096]
    flatten = tf.reshape(
        tensor=layer,
        shape=[BATCH_SIZE, WAV_LENGTH]
    )

    # Input: [64, 4096] > [64, 1]
    logits = tf.layers.dense(
        inputs=flatten,
        units=1
    )[:, 0]

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
