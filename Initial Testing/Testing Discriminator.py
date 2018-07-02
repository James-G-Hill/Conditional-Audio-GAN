import importlib.machinery as im
import numpy as np
import tensorflow as tf
import types

from tensorboard import main as tb


BATCH_SIZE = 64
EPOCHS = None
DISCRIMINATOR = None

MODEL_DIR = 'tmp/testWaveGANDiscriminator'


def createDiscriminator(modelFile):
    """ Creates a discriminator """
    model = _loadModule(
        modelFile,
        '/home/zhanmusi/Dropbox/Birkbeck/' +
        'Advanced Computing Technologies MSc/' +
        'Project/Code/Initial Testing/' + modelFile)
    global DISCRIMINATOR
    DISCRIMINATOR = tf.estimator.Estimator(
        model_fn=model.network,
        model_dir=MODEL_DIR)
    return


def trainDiscriminator(inPath, folders, stepCount):
    """ Trains the discriminator """
    audioLoader = _loadModule(
        'audioDataLoader.py',
        '/home/zhanmusi/Dropbox/Birkbeck/' +
        'Advanced Computing Technologies MSc/' +
        'Project/Code/Audio Manipulation/' + 'audioDataLoader.py')
    data, labels, lookup = audioLoader.loadData(inPath, folders)
    DISCRIMINATOR.train(
        input_fn=_train_input_fn(data, labels),
        steps=stepCount)
    return


def runTensorBoard():
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
