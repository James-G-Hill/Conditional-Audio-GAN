import importlib.util as util
import numpy as np
import tensorflow as tf


BATCH_SIZE = 64
EPOCHS = None
DISCRIMINATOR = None


def createDiscriminator(model):
    """ Creates a discriminator """
    global DISCRIMINATOR
    DISCRIMINATOR = tf.estimator.Estimator(
        model_fn=model,
        model_dir='tmp/testWaveGANDiscriminator')
    return


def trainDiscriminator(inPath, folders, stepCount):
    """ Trains the discriminator """
    audioLoader = _loadModule(
        'audioDataLoader',
        '/home/zhanmusi/Documents/Data/' +
        'Speech Commands Dataset Downsampled/2048')
    data, labels, lookup = audioLoader(inPath, folders)
    DISCRIMINATOR.train(
        input_fn=_train_input_fn(data, labels),
        steps=stepCount)
    return


def _loadModule(modName, modPath):
    """ Loads a module from file location """
    spec = util.spec_from_file_location(modName, modPath)
    module = util.module_from_spec(spec)
    return module


def _train_input_fn(data, labels):
    """ Input function for the custom estimator """
    train_input = tf.estimator.inputs.numpy_input_fn(
        x={'x': np.array(data)},
        y=labels,
        batch_size=BATCH_SIZE,
        num_epochs=EPOCHS,
        shuffle=True)
    return train_input
