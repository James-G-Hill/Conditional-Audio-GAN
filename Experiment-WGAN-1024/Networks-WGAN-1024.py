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


def generator(z):
    """ A waveGAN generator """

    # Input: [64, 100] > [64, 1024]
    densify = tf.layers.dense(
        inputs=z,
        units=WAV_LENGTH,
        name="Z-Input"
    )

    # Input: [64, 1024] > [64, 16, 64]
    shape = tf.reshape(
        tensor=densify,
        shape=[BATCH_SIZE, MODEL_SIZE, MODEL_SIZE * 4]
    )

    relu1 = tf.nn.relu(shape)

    # Input: [64, 16, 64] > [64, 64, 32]
    trans_conv_1 = tf.layers.conv2d_transpose(
        inputs=tf.expand_dims(relu1, axis=1),
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


def discriminator(features):
    """ A waveGAN discriminator """

    # Input: [64, 1024, 1] > [64, 256, 16]
    convolution1 = tf.layers.conv1d(
        inputs=features,
        filters=MODEL_SIZE,
        kernel_size=KERNEL_SIZE,
        strides=STRIDE,
        padding='same',
        use_bias=True,
        activation=tf.nn.leaky_relu,
        # name="Convolution1"
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
        activation=tf.nn.leaky_relu,
        # name="Convolution2"
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
        activation=tf.nn.leaky_relu,
        # name="Convolution3"
    )

    # Input: [64, 16, 64] > [64, 1024]
    flatten = tf.reshape(
        tensor=convolution3,
        shape=[BATCH_SIZE, WAV_LENGTH],
        # name="DiscriminatorFlatten"
    )

    # Input: [64, 1024] > [64, 1]
    logits = tf.layers.dense(
        inputs=flatten,
        units=CLASSES,
        # name='Logits'
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
