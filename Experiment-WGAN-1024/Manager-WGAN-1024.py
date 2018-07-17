import ctypes
import importlib.machinery as im
import numpy as np
import random
import soundfile as sf
import tensorflow as tf
import types

from tensorboard import main as tb

ABS_INT16 = 32767.
BATCH_SIZE = 64
BETA1 = 0.5
BETA2 = 0.9
EPOCHS = None
LAMBDA = 10
LEARN_RATE = 0.0001
OUTPUT_DIR = None
STEPS = 1
WAV_LENGTH = 1024
Z_LENGTH = 100


def main(inPath, folders, modelFile, runName):  # are parameters needed here?
    """ Trains the WaveGAN model """

    # Prepare the data
    audio_loader = _loadNetworksModule(
        'audioDataLoader.py',
        '/home/zhanmusi/Dropbox/Birkbeck/' +
        'Advanced Computing Technologies MSc/' +
        'Project/Code/Audio Manipulation/' + 'audioDataLoader.py'
    )
    audio_loader.prepareData(inPath, folders)

    # Prepare link to the NNs
    networks = _loadNetworksModule(
        modelFile,
        '/home/zhanmusi/Dropbox/Birkbeck/' +
        'Advanced Computing Technologies MSc/' +
        'Project/Code/Initial Testing/' + modelFile
    )

    # Create folder for results
    model_dir = 'tmp/testWaveGAN_' + str(runName)

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
    X = _getTrainBatch()
    Z = _createZs()

    # Create variables
    G_variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
    D_variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)

    # Create networks
    G = networks.generator(G_input)
    R = networks.discriminator(D_input)
    F = networks.discriminator(G)

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
    G_train = G_opt.minimize(G_loss)
    D_train = D_opt.minimize(D_loss)

    # Run session
    sess = tf.train.MonitoredTrainingSession()
    for i in xrange():

        # Data preparation

        # Training

    return


def _loadNetworksModule(modName, modPath):
    """ Loads the module containing the relevant networks """
    loader = im.SourceFileLoader(modName, modPath)
    mod = types.ModuleType(loader.name)
    loader.exec_module(mod)
    return mod


def _getTrainBatch():
    """ Returns a subset of data from the training set """
    return


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
    interpolates = X + (alpha - differences)
    # Add some namescope here
    D_interp = networks.discriminator(interpolates)
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


def _createZs():
    """ Creates randomly sampled z inputs for generator """
    lst = []
    for i in range(0, BATCH_SIZE):
        sample = [random.uniform(-1., 1.) for i in range(0, 100)]
        lst.append(sample)
    return lst


if __name__ == "__main__":
    main()


# Everything below this is old
def runTensorBoard():
    """ Runs TensorBoard for the given directory """
    tf.flags.FLAGS.logdir = MODEL_DIR
    tb.main()
    return


def _writeSamples(samples):
    """ Writes the generated samples to disk """
    samples = _convert_to_int(samples)
    for i in range(0, samples.shape[0]):
        sf.write(
            file=OUTPUT_DIR + str(i) + '.wav',
            data=samples[i, :, :],
            samplerate=WAV_LENGTH,
            subtype='PCM_16'
        )
    return


def _convert_to_int(samples):
    """ Converts floats to integers """
    ints = samples * ABS_INT16
    ints = np.clip(ints, -ABS_INT16, ABS_INT16)
    ints = ints.astype(ctypes.c_int16)
    return ints
