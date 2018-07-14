import importlib.machinery as im
import numpy as np
import tensorflow as tf
import types

from tensorboard import main as tb


AUDIO_LOADER = None
BATCH_SIZE = 64
EPOCHS = None
DISCRIMINATOR = None
MODEL_DIR = None


def createDiscriminator(inPath, folders, modelFile, runName):
    """ Creates a discriminator """
    model = _loadModule(
        modelFile,
        '/home/zhanmusi/Dropbox/Birkbeck/' +
        'Advanced Computing Technologies MSc/' +
        'Project/Code/Initial Testing/' + modelFile
    )
    global AUDIO_LOADER
    global MODEL_DIR
    global DISCRIMINATOR
    AUDIO_LOADER = _loadModule(
        'audioDataLoader.py',
        '/home/zhanmusi/Dropbox/Birkbeck/' +
        'Advanced Computing Technologies MSc/' +
        'Project/Code/Audio Manipulation/' + 'audioDataLoader.py'
    )
    AUDIO_LOADER.prepareData(inPath, folders)
    MODEL_DIR = 'tmp/testWaveGANDiscriminator_' + str(runName)
    DISCRIMINATOR = tf.estimator.Estimator(
        model_fn=model.network,
        model_dir=MODEL_DIR
    )
    return


def trainDiscriminator(stepCount):
    """ Trains the discriminator """
    data, labels = AUDIO_LOADER.loadTrainData()
    DISCRIMINATOR.train(
        input_fn=_train_input_fn(data, labels),
        steps=stepCount
    )
    return


def testDiscriminator(stepCount, name):
    """ Evaluates the discriminator """
    data, labels = AUDIO_LOADER.loadTestData()
    DISCRIMINATOR.evaluate(
        input_fn=_train_input_fn(data, labels),
        steps=stepCount,
        name=str(name)
    )
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
        y={'y': np.array(labels)},
        batch_size=BATCH_SIZE,
        num_epochs=EPOCHS,
        shuffle=True
    )
    return train_input
