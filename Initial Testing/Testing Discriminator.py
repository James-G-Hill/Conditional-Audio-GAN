import importlib.util as util
import tensorflow as tf


def trainDiscriminator(inPath, folders, stepCount):
    """ Trains the discriminator """
    audioLoader = _loadModule(
        'audioDataLoader',
        '/home/zhanmusi/Documents/Data/' +
        'Speech Commands Dataset Downsampled 2048')
    data, labels, lookup = audioLoader(inPath, folders)
    waveGANestimator = tf.estimator.Estimator(
        model_fn=waveGANdiscriminator,
        model_dir='tmp/testWaveGANDiscriminator')
    waveGANestimator.train(
        input_fn=,
        steps=stepCount)
    return


def _loadModule(modName, modPath):
    """ Loads a module from file location """
    spec = util.spec_from_file_location(modName, modPath)
    mod = util.module_from_spec(spec)
    return mod


def _waveGANdiscriminator():
    """ A discriminator for a WaveGAN Model """
    return
