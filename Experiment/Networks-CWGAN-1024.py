import random as rd
import tensorflow as tf


BATCH_SIZE = -1
BETA1 = 0.5
BETA2 = 0.9
CHANNELS = 1
CLASSES = 2
KERNEL_SIZE = 25
LEARN_RATE = 0.0001
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
    densify = tf.layers.dense(
        inputs=x,
        units=WAV_LENGTH - (CLASSES * MODEL_SIZE),
        name="Z-Input"
    )

    # Input: [64, 992] > [64, 16, 62]
    shape = tf.reshape(
        tensor=densify,
        shape=[BATCH_SIZE, 16, (MODEL_SIZE * 4) - 2]
    )

    # Input: [64, 16, 62] > [64, 16, 64]
    concat = tf.concat(values=[shape, y], axis=2)

    # Input: [64, 16, 64] > [64, 64, 32]
    trans_conv_1 = tf.layers.conv2d_transpose(
        inputs=tf.expand_dims(concat, axis=1),
        filters=MODEL_SIZE * 2,
        kernel_size=(1, KERNEL_SIZE),
        strides=(1, STRIDE),
        padding='SAME',
        name="TransConvolution1"
    )[:, 0]

    relu2 = tf.nn.relu(trans_conv_1)

    # Input: [64, 64, 32] > [64, 256, 16]
    trans_conv_2 = tf.layers.conv2d_transpose(
        inputs=tf.expand_dims(relu2, axis=1),
        filters=MODEL_SIZE,
        kernel_size=(1, KERNEL_SIZE),
        strides=(1, STRIDE),
        padding='SAME',
        name="TransConvolution2"
    )[:, 0]

    relu3 = tf.nn.relu(trans_conv_2)

    # Input: [64, 256, 16] > [64, 1024, 1]
    trans_conv_3 = tf.layers.conv2d_transpose(
        inputs=tf.expand_dims(relu3, axis=1),
        filters=CHANNELS,
        kernel_size=(1, KERNEL_SIZE),
        strides=(1, STRIDE),
        padding='SAME',
        name="TransConvolution3"
    )[:, 0]

    # Input: [64, 1024, 1]
    tanh = tf.tanh(
        x=trans_conv_3,
        name="GeneratedSamples"
    )

    return tanh


def discriminator(x, y):
    """ A waveGAN discriminator """

    concat = tf.concat(values=[x, y], axis=2)

    # Input: [64, 1024, 3] > [64, 256, 16]
    convolution1 = tf.layers.conv1d(
        inputs=concat,
        filters=MODEL_SIZE,
        kernel_size=KERNEL_SIZE,
        strides=STRIDE,
        padding='same',
        use_bias=True,
        activation=tf.nn.leaky_relu
    )

    convolution1 = _phaseShuffle(convolution1)

    # Input: [64, 256, 16] > [64, 64, 32]
    convolution2 = tf.layers.conv1d(
        inputs=convolution1,
        filters=MODEL_SIZE * 2,
        kernel_size=KERNEL_SIZE,
        strides=STRIDE,
        padding='same',
        use_bias=True,
        activation=tf.nn.leaky_relu
    )

    convolution2 = _phaseShuffle(convolution2)

    # Input: [64, 64, 32] > [64, 16, 64]
    convolution3 = tf.layers.conv1d(
        inputs=convolution2,
        filters=MODEL_SIZE * 4,
        kernel_size=KERNEL_SIZE,
        strides=STRIDE,
        padding='same',
        use_bias=True,
        activation=tf.nn.leaky_relu
    )

    # Input: [64, 16, 64] > [64, 1024]
    flatten = tf.reshape(
        tensor=convolution3,
        shape=[BATCH_SIZE, WAV_LENGTH]
    )

    # Input: [64, 1024] > [64, 1]
    logits = tf.layers.dense(
        inputs=flatten,
        units=1
    )

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


def _returnPhaseShuffleValue():
    """ Returns a a ranom integer in the range decided for phase shuffle"""
    return rd.randint(-PHASE_SHUFFLE, PHASE_SHUFFLE)
