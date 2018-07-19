import argparse as ag
import importlib.machinery as im
import tensorflow as tf
import types

ABS_INT16 = 32767.
BATCH_SIZE = 64
BETA1 = 0.5
BETA2 = 0.9
D_UPDATES_PER_G_UPDATES = 1
EPOCHS = None
LAMBDA = 10
LEARN_RATE = 0.0001
NETWORKS = None
OUTPUT_DIR = None
RUNS = 20
STEPS = 1
WAV_LENGTH = 1024
Z_LENGTH = 100

DATA_PATH = "/home/zhanmusi/Documents/Data/"
DOWN_PATH = "Speech Commands Dataset Downsampled/"
TRAIN_DATA_FOLDER = DATA_PATH + DOWN_PATH + str(WAV_LENGTH) + "/"

MODEL_DIR = None


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
        '/home/zhanmusi/Dropbox/Birkbeck/' +
        'Advanced Computing Technologies MSc/' +
        'Project/Code/Audio Manipulation/' +
        'audioDataLoader.py'
    )
    audio_loader.prepareData(TRAIN_DATA_FOLDER, folders)

    # Prepare link to the NNs
    global NETWORKS
    NETWORKS = _loadNetworksModule(
        'Networks-WGAN-' + str(WAV_LENGTH) + '.py',
        '/home/zhanmusi/Dropbox/Birkbeck/' +
        'Advanced Computing Technologies MSc/' +
        'Project/Code/Experiment-WGAN-' + str(WAV_LENGTH) + '/' +
        'Networks-WGAN-' + str(WAV_LENGTH) + '.py'
    )

    # Create folder for results
    global MODEL_DIR
    MODEL_DIR = 'tmp/testWaveGAN_' + str(WAV_LENGTH) + '_' + runName[0]

    # Create input placeholder
    G_input = tf.placeholder(
        tf.float32,
        shape=[None, Z_LENGTH],
        name='Noise'
    )
    D_input = tf.placeholder(
        tf.float32,
        shape=[None, WAV_LENGTH],
        name='Waves'
    )

    # Create data
    Z = tf.random_uniform([BATCH_SIZE, Z_LENGTH], -1., 1., dtype=tf.float32)
    X = audio_loader.loadTestData()

    # Create variables
    G_variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
    D_variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)

    # Create networks
    G = NETWORKS.generator(G_input)
    R = NETWORKS.discriminator(D_input)
    F = NETWORKS.discriminator(G)

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
        checkpoint_dir=MODEL_DIR + '/Checkpoint',
        save_checkpoint_secs=300,
        save_summaries_secs=120)
    for _ in range(RUNS):
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
        default=['zero', 'one'],
        help="The words for sounds you want to train with."
    )
    main(parser.parse_args())
