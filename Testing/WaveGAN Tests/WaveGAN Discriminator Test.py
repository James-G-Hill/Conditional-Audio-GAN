import librosa as lb
import numpy as np
import os
import tensorflow as tf

# Discriminator constants
BATCH_SIZE = 64
CHANNELS = 1
FILTER_LENGTH = 25
LEARN_RATE = 0.001
MODEL_SIZE = 64
STRIDE = 4
WAV_LENGTH = 4096

# Loading constants
FOLDER_NAMES = ['zero', 'one']

# Training constants
STEPS = 1
EPOCHS = None


def waveGANdiscriminator(features, labels, mode):
    """ A discriminator for a WaveGAN Model """

    print(features['x'])

    # Input layer
    input_layer = tf.reshape(
        tensor=tf.cast(features['x'], tf.float32),
        shape=[BATCH_SIZE, WAV_LENGTH, CHANNELS],
        name='input_layer')

    print(input_layer)

    # 1st convolution layer
    conv1 = tf.layers.conv1d(
        inputs=input_layer,
        filters=MODEL_SIZE,
        kernel_size=25,  # [FILTER_LENGTH, CHANNELS, MODEL_SIZE],
        strides=STRIDE,
        padding='same',
        use_bias=True,
        activation=tf.nn.leaky_relu,
        name="conv1")

    print(conv1)

    # 1st phase shuffle

    # 2nd convolution layer
    conv2 = tf.layers.conv1d(
        inputs=conv1,
        filters=MODEL_SIZE * 2,
        kernel_size=25,  # [FILTER_LENGTH, MODEL_SIZE, 2 * MODEL_SIZE],
        strides=STRIDE,
        padding='same',
        use_bias=True,
        activation=tf.nn.leaky_relu,
        name="conv2")

    print(conv2)

    # 2nd phase shuffle

    # 3rd convolution layer
    conv3 = tf.layers.conv1d(
        inputs=conv2,
        filters=MODEL_SIZE * 4,
        kernel_size=25,  # [FILTER_LENGTH, 2 * MODEL_SIZE, 4 * MODEL_SIZE],
        strides=STRIDE,
        padding='same',
        use_bias=True,
        activation=tf.nn.leaky_relu,
        name="conv3")

    print(conv3)

    # 3rd phase shuffle

    # 4th convolution layer
    conv4 = tf.layers.conv1d(
        inputs=conv3,
        filters=MODEL_SIZE * 8,
        kernel_size=25,  # [FILTER_LENGTH, 4 * MODEL_SIZE, 8 * MODEL_SIZE],
        strides=STRIDE,
        padding='same',
        use_bias=True,
        activation=tf.nn.leaky_relu,
        name="conv4")

    print(conv4)

    # 4th phase shuffle

    # 5th convolution layer
    conv5 = tf.layers.conv1d(
        inputs=conv4,
        filters=MODEL_SIZE * 16,
        kernel_size=25,  # [FILTER_LENGTH, 8 * MODEL_SIZE, 16 * MODEL_SIZE],
        strides=STRIDE,
        padding='same',
        use_bias=True,
        activation=tf.nn.leaky_relu,
        name="conv5")

    print(conv5)

    # Reshape
    reshape = tf.reshape(
        tensor=conv5,
        shape=[BATCH_SIZE, 64 * MODEL_SIZE],  # unsure what 64 represents here
        name='reshape')

    print(reshape)

    # Dense
    result = tf.layers.dense(
        inputs=reshape,
        units=64 * MODEL_SIZE,  # Seems to be same '64' as above
        name='dense'
    )

    print(result)

    predictions = {
        'classes': tf.argmax(input=result, axis=1),
        'probabilities': tf.nn.softmax(result, name='softmax_tensor')
    }

    loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=result)

    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.train.GradientDescentOptimizer(
            learning_rate=LEARN_RATE)
        train_op = optimizer.minimize(
            loss=loss,
            global_step=tf.train.get_global_step())
        return tf.estimator.EstimatorSpec(
            mode=mode,
            loss=loss,
            train_op=train_op)

    eval_metric_ops = {
        'accuracy': tf.metrics.accuracy(
            labels=labels,
            predictions=predictions['classes'])}

    return tf.estimator.EstimatorSpec(
        mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)


def loadData():
    """ Loads the audio data into list form """

    # trainData = dict()
    trainData = []
    trainLabels = []
    outFilePath = \
        '/home/zhanmusi/Documents/Data/Speech Commands Dataset Downsampled/'

    for folder in FOLDER_NAMES:
        if folder == 'zero':  # creating labels as numbers needs fixing
            label = 0
        if folder == 'one':
            label = 1
        files = lb.util.find_files(outFilePath + folder + '/', ext='wav')
        for file in files:
            path, name = os.path.split(file)
            series, sampRate = lb.core.load(file, sr=None)
            if len(series) < WAV_LENGTH:
                series = np.append(series, np.zeros(WAV_LENGTH - len(series)))
            trainData.append(series)
            trainLabels.append(label)

    # trainData = tf.cast(trainData, tf.float32)
    trainLabels = np.array(trainLabels)
    return trainData, trainLabels


def train_input_fn():
    """ Input function for the custom estimator """

    train_data, train_labels = loadData()

    train_input = tf.estimator.inputs.numpy_input_fn(
        x={'x': np.array(train_data)},
        y=train_labels,
        batch_size=BATCH_SIZE,
        num_epochs=EPOCHS,
        shuffle=True)

    return train_input


def trainDiscriminator():
    """ Trains the discriminator """

    waveGANclassifier = tf.estimator.Estimator(
        model_fn=waveGANdiscriminator,
        model_dir='tmp/testWaveGANDiscriminator')

    waveGANclassifier.train(
        input_fn=train_input_fn(),
        steps=STEPS)

    return waveGANclassifier


# exec(open('WaveGAN Discriminator Test.py').read())
