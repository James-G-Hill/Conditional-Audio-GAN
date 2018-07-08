import tensorflow as tf

BATCH_SIZE = 64
CHANNELS = 1
KERNEL_SIZE = 25
MODEL_SIZE = 16
STRIDE = [1, 4, 1]
WAV_LENGTH = 1024
Z_LENGTH = 100


def generate(z_input):
    """ A waveGAN generator """

    # Input: [64, 100] > [64, 1024]
    densify = tf.layers.dense(
        inputs=z_input,
        units=[Z_LENGTH * WAV_LENGTH],
        name="Input Dense"
    )

    # Input: [64, 1024] > [64, 16, 64]
    shape = tf.reshape(
        tensor=tf.cast(densify, tf.float32),
        shape=[BATCH_SIZE, MODEL_SIZE, MODEL_SIZE * 4]
    )

    relu = tf.nn.relu(shape)

    # Input: [64, 16, 64] > [64, 64, 32]
    trans_conv_1 = tf.contrib.nn.conv1_transpose(
        value=relu,
        filter=[KERNEL_SIZE, MODEL_SIZE * 2, MODEL_SIZE * 4],
        output_shape=MODEL_SIZE * 4,
        stride=STRIDE,
        padding='same',
        name="Trans_Convolution_1"
    )

    relu = tf.nn.relus(trans_conv_1)

    # Input: [64, 64, 32] > [64, 256, 16]
    trans_conv_2 = tf.contrib.nn.conv1_transpose(
        value=relu,
        filter=[KERNEL_SIZE, MODEL_SIZE * 1, MODEL_SIZE * 2],
        output_shape=MODEL_SIZE * 16,
        stride=STRIDE,
        padding='same',
        name="Trans_Convolution_2"
    )

    relu = tf.nn.relu(trans_conv_2)

    # Input: [64, 256, 16] > [64, 1024, 1]
    trans_conv_3 = tf.contrib.nn.conv1_transpose(
        value=relu,
        filter=[KERNEL_SIZE, CHANNELS, MODEL_SIZE * 1],
        output_shape=MODEL_SIZE * 16,
        stride=STRIDE,
        padding='same',
        name="Trans_Convolution_3"
    )

    # Input: [64, 1024, 1]
    activation_tanh = tf.tanh(
        x=trans_conv_3,
        name="Final Tanh"
    )

    return activation_tanh
