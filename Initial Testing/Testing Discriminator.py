import importlib.util as util
import numpy as np
import tensorflow as tf


BATCH_SIZE = 64
EPOCHS = None


def trainDiscriminator(inPath, folders, stepCount):
    """ Trains the discriminator """
    audioLoader = _loadModule(
        'audioDataLoader',
        '/home/zhanmusi/Documents/Data/' +
        'Speech Commands Dataset Downsampled 2048')
    data, labels, lookup = audioLoader(inPath, folders)
    waveGANestimator = tf.estimator.Estimator(
        model_fn=_waveGANdiscriminator,
        model_dir='tmp/testWaveGANDiscriminator')
    waveGANestimator.train(
        input_fn=_train_input_fn(data, labels),
        steps=stepCount)
    return


def _train_input_fn(data, labels):
    """ Input function for the custom estimator """
    train_input = tf.estimator.inputs.numpy_input_fn(
        x={'x': np.array(data)},
        y=labels,
        batch_size=BATCH_SIZE,
        num_epochs=EPOCHS,
        shuffle=True)
    return train_input


def _loadModule(modName, modPath):
    """ Loads a module from file location """
    spec = util.spec_from_file_location(modName, modPath)
    mod = util.module_from_spec(spec)
    return mod


def _waveGANdiscriminator():
    """ A discriminator for a WaveGAN Model """
    return
