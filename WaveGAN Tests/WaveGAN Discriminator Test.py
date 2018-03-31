import tensorflow as tf

BATCH_SIZE = 64
CHANNELS = 1
FILTER_LENGTH = 25
MODEL_SIZE = 64
STRIDE = 4
WAV_LENGTH = 4096


def waveGANdiscriminator(wav):
    """ A discriminator for a WaveGAN Model """

    # Input layer
    input_layer = tf.reshape(
        tensor=wav,
        shape=[BATCH_SIZE, WAV_LENGTH, CHANNELS],
        name='input_layer')

    # 1st convolution layer
    conv1 = tf.layers.conv1d(
        inputs=input_layer,
        kernel_size=[FILTER_LENGTH, CHANNELS, MODEL_SIZE],
        strides=STRIDE,
        use_bias=True,
        activation=tf.nn.leaky_relu,
        name="conv1")

    # 1st phase shuffle

    # 2nd convolution layer
    conv2 = tf.layers.conv1d(
        inputs=conv1,
        kernel_size=[FILTER_LENGTH, MODEL_SIZE, 2 * MODEL_SIZE],
        strides=STRIDE,
        use_bias=True,
        activation=tf.nn.leaky_relu,
        name="conv2")

    # 2nd phase shuffle

    # 3rd convolution layer
    conv3 = tf.layers.conv1d(
        inputs=conv2,
        kernel_size=[FILTER_LENGTH, 2 * MODEL_SIZE, 4 * MODEL_SIZE],
        strides=STRIDE,
        use_bias=True,
        activation=tf.nn.leaky_relu,
        name="conv3")

    # 3rd phase shuffle

    # 4th convolution layer
    conv4 = tf.layers.conv1d(
        inputs=conv3,
        kernel_size=[FILTER_LENGTH, 4 * MODEL_SIZE, 8 * MODEL_SIZE],
        strides=STRIDE,
        use_bias=True,
        activation=tf.nn.leaky_relu,
        name="conv4")

    # 4th phase shuffle

    # 5th convolution layer
    conv5 = tf.layers.conv1d(
        inputs=conv4,
        kernel_size=[FILTER_LENGTH, 8 * MODEL_SIZE, 16 * MODEL_SIZE],
        strides=STRIDE,
        use_bias=True,
        activation=tf.nn.leaky_relu,
        name="conv5")

    # Reshape
    reshape = tf.reshape(
        tensor=conv5,
        shape=[BATCH_SIZE, 256 * MODEL_SIZE],
        name='reshape')

    # Dense
    result = tf.layers.dense(
        inputs=reshape,
        units=[BATCH_SIZE, 1]
    )

    return result
