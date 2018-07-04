import tensorflow as tf

BATCH_SIZE = 64
BETA1 = 0.5
BETA2 = 0.9
CHANNELS = 1
LEARN_RATE = 0.0001
MODEL_SIZE = 64
PHASE_SHUFFLE = 2
STRIDE = [BATCH_SIZE, 4, CHANNELS]
WAV_LENGTH = 2048


def network(features, labels, mode):
    """ A waveGAN discriminator """

    inputLayer = tf.reshape(
        tensor=tf.cast(features['x'], tf.float32),
        shape=[BATCH_SIZE, WAV_LENGTH, CHANNELS],
        name='InputLayer'
    )

    # Input: 64, 2048, 1
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

    # Input: 64, 512, 64
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

    # Input: 64, 128, 128
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

    # Input: 64, 32,
    reshape = tf.reshape(
        tensor=convolution3,
        shape=[BATCH_SIZE, 32 * MODEL_SIZE],
        name="Reshape"
    )

    # Input: 64,
    result = tf.layers.dense(
        inputs=reshape,
        units=BATCH_SIZE * 1,
        name='dense')

    loss = tf.losses.sparse_softmax_cross_entropy(
        labels=labels,
        logits=result
    )

    optimizer = tf.train.AdamOptimizer(
        learning_rate=LEARN_RATE,
        beta1=BETA1,
        beta2=BETA2
    )

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


def _phaseShuffle(layers):
    """ Shuffles the phase of each layer """
    return
