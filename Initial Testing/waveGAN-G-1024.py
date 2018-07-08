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

    relu = tf.nn.relu(shape)

    # Input: [64, 16, 64] > [64, 64, 32]
    trans_conv_1 = tf.contrib.nn.conv1d_transpose(
        value=relu,
        filter=tf.constant(
            0,
            dtype=tf.float32,
            shape=[KERNEL_SIZE, MODEL_SIZE * 2, MODEL_SIZE * 4]),
        output_shape=[BATCH_SIZE, MODEL_SIZE * 4, MODEL_SIZE * 2],
        stride=STRIDE,
        padding='SAME',
        name="Trans_Convolution_1"
    )

    relu = tf.nn.relu(trans_conv_1)

    # Input: [64, 64, 32] > [64, 256, 16]
    trans_conv_2 = tf.contrib.nn.conv1d_transpose(
        value=relu,
        filter=tf.constant(
            0,
            dtype=tf.float32,
            shape=[KERNEL_SIZE, MODEL_SIZE, MODEL_SIZE * 2]),
        output_shape=[BATCH_SIZE, MODEL_SIZE * 16, MODEL_SIZE],
        stride=STRIDE,
        padding='SAME',
        name="Trans_Convolution_2"
    )

    relu = tf.nn.relu(trans_conv_2)

    # Input: [64, 256, 16] > [64, 1024, 1]
    trans_conv_3 = tf.contrib.nn.conv1d_transpose(
        value=relu,
        filter=tf.constant(
            0,
            dtype=tf.float32,
            shape=[KERNEL_SIZE, CHANNELS, MODEL_SIZE * 1]),
        output_shape=[BATCH_SIZE, MODEL_SIZE * 64, CHANNELS],
        stride=STRIDE,
        padding='SAME',
        name="Trans_Convolution_3"
    )

    # Input: [64, 1024, 1]
    tanh = tf.tanh(
        x=trans_conv_3,
        name="Final_Tanh"
    )

    return tanh
