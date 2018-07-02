import tensorflow as tf

BATCH_SIZE = 64
CHANNELS = 1
LEARN_RATE = 0.001
MODEL_SIZE = 64
STRIDE = 4
WAV_LENGTH = 2048


def network(features, labels, mode):
    """ A waveGAN discriminator """

    inputLayer = tf.reshape(
        tensor=tf.cast(features['x'], tf.float32),
        shape=[BATCH_SIZE, WAV_LENGTH, CHANNELS],
        name='InputLayer'
    )

    convolution1 = tf.layers.conv1d(
        inputs=inputLayer,
        filters=MODEL_SIZE,
        kernel_size=25,
        strides=STRIDE,
        padding='same',
        use_bias=True,
        activation=tf.nn.leaky_relu,
        name="Convolution1"
    )

    convolution2 = tf.layers.conv1d(
        inputs=convolution1,
        filters=MODEL_SIZE * 2,
        kernel_size=25,
        strides=STRIDE,
        padding='same',
        use_bias=True,
        activation=tf.nn.leaky_relu,
        name="Convolution2"
    )

    convolution3 = tf.layers.conv1d(
        inputs=convolution2,
        filters=MODEL_SIZE * 4,
        kernel_size=25,
        strides=STRIDE,
        padding='same',
        use_bias=True,
        activation=tf.nn.leaky_relu,
        name="Convolution3"
    )

    convolution4 = tf.layers.conv1d(
        inputs=convolution3,
        filters=MODEL_SIZE * 8,
        kernel_size=25,
        strides=STRIDE,
        padding='same',
        use_bias=True,
        activation=tf.nn.leaky_relu,
        name="Convolution4"
    )

    convolution5 = tf.layers.conv1d(
        inputs=convolution4,
        filters=MODEL_SIZE * 16,
        kernel_size=25,
        strides=STRIDE,
        padding='same',
        use_bias=True,
        activation=tf.nn.leaky_relu,
        name="Convolution5"
    )

    reshape = tf.reshape(
        tensor=convolution5,
        shape=[BATCH_SIZE, 32 * MODEL_SIZE],
        name="Reshape"
    )

    result = tf.layers.dense(
        inputs=reshape,
        units=64 * MODEL_SIZE,
        name='dense')

    loss = tf.losses.sparse_softmax_cross_entropy(
        labels=labels,
        logits=result
    )

    optimizer = tf.train.GradientDescentOptimizer(learning_rate=LEARN_RATE)

    train_op_param = optimizer.minimize(
        loss=loss,
        global_step=tf.train.get_global_step()
    )

    predictions = {
        'classes': tf.argmax(input=result, axis=1),
        'probabilities': tf.nn.softmax(result, name='softmax_tensor')
    }

    eval_metrics = {
        'accuracy': tf.metrics.accuracy(
            labels=labels,
            predictions=predictions['classes'])
    }

    estimator = tf.estimator.EstimatorSpec(
        eval_metric_ops=eval_metrics,
        loss=loss,
        train_op=train_op_param,
        mode=mode
    )

    return estimator
