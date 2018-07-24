import argparse as ag
import importlib.machinery as im
import os
import numpy as np
import tensorflow as tf
import types

ABS_INT16 = 32767.
BATCH_SIZE = 64
BETA1 = 0.5
BETA2 = 0.9
D_UPDATES_PER_G_UPDATES = 1
LAMBDA = 10
LEARN_RATE = 0.0001
NETWORKS = None
OUTPUT_DIR = None
RUNS = 200000
WAV_LENGTH = 1024
Z_LENGTH = 100


def main(args):
    """ Runs the relevant command passed through arguments """
    if args.mode == "train":
        _train(args.words, args.runName)
    return


def _train(folders, runName):
    """ Trains the WaveGAN model """

    # Prepare the data
    audio_loader = _loadNetworksModule(
        'audioDataLoader.py',
        os.path.abspath(
            os.path.join(
                os.path.dirname((__file__)),
                os.pardir,
                'Audio Manipulation/',
                'audioDataLoader.py'
            )
        )
    )
    training_data_path = os.path.abspath(
        os.path.join(
            os.path.dirname((__file__)),
            os.pardir,
            "Speech Commands Dataset Downsampled/",
            str(WAV_LENGTH)
        )
    )
    audio_loader.prepareData(training_data_path, folders)

    # Prepare link to the NNs
    global NETWORKS
    NETWORKS = _loadNetworksModule(
        'Networks-WGAN-' + str(WAV_LENGTH) + '.py',
        'Networks-WGAN-' + str(WAV_LENGTH) + '.py'
    )

    # Create folder for results
    model_dir = 'tmp/testWaveGAN_' + str(WAV_LENGTH) + '_' + runName[0]

    # Create data
    Z = tf.random_uniform([BATCH_SIZE, Z_LENGTH], -1., 1., dtype=tf.float32)
    X, X_labels = audio_loader.loadAllData()
    X_length = len(X)
    X = np.vstack(X)
    X = tf.reshape(
        tensor=tf.cast(X, tf.float32),
        shape=[X_length, WAV_LENGTH, 1]
    )
    X = tf.data.Dataset.from_tensor_slices(X)
    X = X.shuffle(buffer_size=X_length)
    X = X.apply(tf.contrib.data.batch_and_drop_remainder(BATCH_SIZE))
    X = X.repeat()
    X = X.make_one_shot_iterator()
    X = X.get_next()

    # Create networks
    with tf.variable_scope('G'):
        G = NETWORKS.generator(Z)
    with tf.variable_scope('D'):
        R = NETWORKS.discriminator(X)
    with tf.variable_scope('D', reuse=True):
        F = NETWORKS.discriminator(G)

    # Create variables
    G_variables = tf.get_collection(
        tf.GraphKeys.TRAINABLE_VARIABLES,
        scope='G'
    )
    D_variables = tf.get_collection(
        tf.GraphKeys.TRAINABLE_VARIABLES,
        scope='D'
    )

    # Build loss
    G_loss, D_loss = _loss(G, R, F, X, Z)

    # Build optimizers
    G_opt = tf.train.AdamOptimizer(
        learning_rate=LEARN_RATE,
        beta1=BETA1,
        beta2=BETA2
    )
    D_opt = tf.train.AdamOptimizer(
        learning_rate=LEARN_RATE,
        beta1=BETA1,
        beta2=BETA2
    )

    # Build training operations
    G_train_op = G_opt.minimize(
        G_loss,
        var_list=G_variables,
        global_step=tf.train.get_or_create_global_step()
    )
    D_train_op = D_opt.minimize(
        D_loss,
        var_list=D_variables
    )

    # Root Mean Square
    Z_rms = tf.sqrt(tf.reduce_mean(tf.square(G[:, :, 0]), axis=1))
    X_rms = tf.sqrt(tf.reduce_mean(tf.square(X[:, :, 0]), axis=1))

    # Summary
    tf.summary.audio('X', X, WAV_LENGTH)
    tf.summary.audio('G', G, WAV_LENGTH)
    tf.summary.histogram('Z_rms', Z_rms)
    tf.summary.histogram('X_rms', X_rms)
    tf.summary.scalar('G_loss', G_loss)
    tf.summary.scalar('D_loss', D_loss)

    # Run session
    sess = tf.train.MonitoredTrainingSession(
        checkpoint_dir=model_dir + '/Checkpoint',
        config=tf.ConfigProto(log_device_placement=False),
        save_checkpoint_secs=300,
        save_summaries_secs=120)
    for i in range(RUNS):
        if i % 1000 == 0:
            print('Completed Run Number: ' + str(i + 1))
        for _ in range(D_UPDATES_PER_G_UPDATES):
            sess.run(D_train_op)
        sess.run(G_train_op)

    return


def _loadNetworksModule(modName, modPath):
    """ Loads the module containing the relevant networks """
    loader = im.SourceFileLoader(modName, modPath)
    mod = types.ModuleType(loader.name)
    loader.exec_module(mod)
    return mod


def _loss(G, R, F, X, Z):
    """ Calculates the loss """
    G_loss = -tf.reduce_mean(F)
    D_loss = tf.reduce_mean(F) - tf.reduce_mean(R)
    alpha = tf.random_uniform(
        shape=[BATCH_SIZE, 1, 1],
        minval=0.,
        maxval=1.
    )
    differences = G - X
    interpolates = X + (alpha * differences)
    with tf.name_scope('D_interp'), tf.variable_scope('D', reuse=True):
        D_interp = NETWORKS.discriminator(interpolates)
    gradients = tf.gradients(D_interp, [interpolates])[0]
    slopes = tf.sqrt(
        tf.reduce_sum(
            tf.square(gradients),
            reduction_indices=[1, 2]
        )
    )
    gradient_penalty = tf.reduce_mean((slopes - 1.) ** 2.)
    D_loss += LAMBDA * gradient_penalty
    return G_loss, D_loss


if __name__ == "__main__":
    parser = ag.ArgumentParser()
    parser.add_argument(
        dest='runName',
        nargs=1,
        type=str,
        help="A name for this run of the experiment."
    )
    parser.add_argument(
        dest='mode',
        nargs='?',
        type=str,
        default='train',
        help="How wish to use the model."
    )
    parser.add_argument(
        dest='words',
        nargs='*',
        type=str,
        default=['zero'],
        help="The words for sounds you want to train with."
    )
    main(parser.parse_args())
