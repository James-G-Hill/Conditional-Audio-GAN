import ctypes
import importlib.machinery as im
import numpy as np
import random
import soundfile as sf
import tensorflow as tf
import types

ABS_INT16 = 32767.
BATCH_SIZE = 64
GENERATOR = None
WAV_LENGTH = None

MODEL_DIR = None
OUTPUT_DIR = None


def createGenerator(modelFile, wave_length):
    """ Creates a generator """
    model = _loadModule(
        modelFile,
        '/home/zhanmusi/Dropbox/Birkbeck/' +
        'Advanced Computing Technologies MSc/' +
        'Project/Code/Initial Testing/' + modelFile
    )
    global WAV_LENGTH
    WAV_LENGTH = wave_length
    global GENERATOR
    GENERATOR = model
    global OUTPUT_DIR
    OUTPUT_DIR = '/home/zhanmusi/Documents/Data/Generated Samples/' \
                 + str(WAV_LENGTH) + '/'
    return


def generateSamples(batches):
    """ Generates samples """
    for _ in range(0, batches):
        z = tf.placeholder(tf.float32, shape=[BATCH_SIZE, 100])
        G = GENERATOR.generate(z)
        sess = tf.InteractiveSession()
        tf.global_variables_initializer().run()
        z_samples = _createZs()
        samples = sess.run(
            G,
            feed_dict={z: z_samples}
        )
        sess.close()
        _writeSamples(samples)
    return


def _createZs():
    """ Creates randomly sampled z inputs for generator """
    lst = []
    for i in range(0, BATCH_SIZE):
        sample = [random.uniform(-1., 1.) for i in range(0, 100)]
        lst.append(sample)
    return lst


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


def _loadModule(modName, modPath):
    """ Loads a module from file location """
    loader = im.SourceFileLoader(modName, modPath)
    mod = types.ModuleType(loader.name)
    loader.exec_module(mod)
    return mod


def _convert_to_int(samples):
    """ Converts floats to integers """
    ints = samples * ABS_INT16
    ints = np.clip(ints, -ABS_INT16, ABS_INT16)
    ints = ints.astype(ctypes.c_int16)
    return ints

    return
