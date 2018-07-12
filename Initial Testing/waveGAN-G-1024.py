import tensorflow as tf


BATCH_SIZE = 64
CHANNELS = 1
KERNEL_SIZE = 25
MODEL_SIZE = 16
STRIDE = 4
WAV_LENGTH = 1024
Z_LENGTH = 100


def generate(z):
    """ A waveGAN generator """

    # Input: [64, 100] > [64, 1024]
    densify = tf.layers.dense(
        inputs=z,
        units=WAV_LENGTH,
        name="Input_Dense"
    )

    # Input: [64, 1024] > [64, 16, 64]
    shape = tf.reshape(
        tensor=densify,
        shape=[BATCH_SIZE, MODEL_SIZE, MODEL_SIZE * 4]
    )

    # shape = tf.layers.batch_normalization(shape)

    relu1 = tf.nn.relu(shape)

    # Input: [64, 16, 64] > [64, 64, 32]
    trans_conv_1 = tf.layers.conv2d_transpose(
        inputs=tf.expand_dims(relu1, axis=1),
        filters=MODEL_SIZE * 2,
        kernel_size=(1, KERNEL_SIZE),
        strides=(1, STRIDE),
        padding='SAME',
        name="Trans_Convolution_1"
    )[:, 0]

    # trans_conv_1 = tf.layers.batch_normalization(trans_conv_1)

    relu2 = tf.nn.relu(trans_conv_1)

    trans_conv_2 = tf.layers.conv2d_transpose(
        inputs=tf.expand_dims(relu2, axis=1),
        filters=MODEL_SIZE,
        kernel_size=(1, KERNEL_SIZE),
        strides=(1, STRIDE),
        padding='SAME',
        name="Trans_Convolution_2"
    )[:, 0]

    relu3 = tf.nn.relu(trans_conv_2)

    trans_conv_3 = tf.layers.conv2d_transpose(
        inputs=tf.expand_dims(relu3, axis=1),
        filters=CHANNELS,
        kernel_size=(1, KERNEL_SIZE),
        strides=(1, STRIDE),
        padding='SAME',
        name="Trans_Convolution_3"
    )[:, 0]

    # Input: [64, 1024, 1]
    tanh = tf.tanh(
        x=trans_conv_3,
        name="Final_Tanh"
    )

    return tanh
