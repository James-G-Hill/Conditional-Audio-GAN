import importlib.machinery as im
import numpy as np
import tensorflow as tf
import types

from tensorboard import main as tb


BATCH_SIZE = 64
EPOCHS = None
DISCRIMINATOR = None

MODEL_DIR = 'tmp/testWaveGANDiscriminator'

AUDIO_LOADER = None


def createDiscriminator(inPath, folders, modelFile):
    """ Creates a discriminator """
    model = _loadModule(
        modelFile,
        '/home/zhanmusi/Dropbox/Birkbeck/' +
        'Advanced Computing Technologies MSc/' +
        'Project/Code/Initial Testing/' + modelFile)
    AUDIO_LOADER = _loadModule(
        'audioDataLoader.py',
        '/home/zhanmusi/Dropbox/Birkbeck/' +
        'Advanced Computing Technologies MSc/' +
        'Project/Code/Audio Manipulation/' + 'audioDataLoader.py')
    AUDIO_LOADER.prepareData(inPath, folders)
    global DISCRIMINATOR
    DISCRIMINATOR = tf.estimator.Estimator(
        model_fn=model.network,
        model_dir=MODEL_DIR)
    return


def trainDiscriminator(stepCount):
    """ Trains the discriminator """
    data, labels = AUDIO_LOADER.loadTrainData()
    DISCRIMINATOR.train(
        input_fn=_train_input_fn(data, labels),
        steps=stepCount)
    return


def testDiscriminator(data, labels, stepCount, name):
    """ Evaluates the discriminator """
    data, labels = AUDIO_LOADER.loadTestData()
    DISCRIMINATOR.evaluate(
        input_fn=_train_input_fn(data, labels),
        steps=stepCount,
        name=name)
    return


def runTensorBoard():
    """ Runs TensorBoard for the given directory """
    tf.flags.FLAGS.logdir = MODEL_DIR
    tb.main()
    return


def _loadModule(modName, modPath):
    """ Loads a module from file location """
    loader = im.SourceFileLoader(modName, modPath)
    mod = types.ModuleType(loader.name)
    loader.exec_module(mod)
    return mod


def _train_input_fn(data, labels):
    """ Input function for the custom estimator """
    train_input = tf.estimator.inputs.numpy_input_fn(
        x={'x': np.array(data)},
        y=np.array(labels),
        batch_size=BATCH_SIZE,
        num_epochs=EPOCHS,
        shuffle=True)
    return train_input
