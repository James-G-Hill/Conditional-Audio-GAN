import ctypes
import importlib.machinery as im
import numpy as np
import random
import soundfile as sf
import tensorflow as tf
import types

from tensorboard import main as tb


ABS_INT16 = 32767.
AUDIO_LOADER = None
BATCH_SIZE = 64
EPOCHS = None
DISCRIMINATOR = None
GENERATOR = None
MODEL_DIR = None
OUTPUT_DIR = None
WAV_LENGTH = None


def main():
    return


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


def testDiscriminator(name):
    """ Evaluates the discriminator """
    data, labels = AUDIO_LOADER.loadTestData()
    results = DISCRIMINATOR.evaluate(
        input_fn=_eval_input_fn(data, labels),
        name=str(name)
    )
    print(results)
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
        shuffle=True
    )
    return train_input


def _eval_input_fn(data, labels):
    """ Input function for the custom estimator """
    eval_input = tf.estimator.inputs.numpy_input_fn(
        x={'x': np.array(data)},
        y=np.array(labels),
        num_epochs=1,
        shuffle=False
    )
    return eval_input


def createGenerator(modelFile, wave_length):
    """ Creates a generator """
    model = _loadModule(
        modelFile,
        '/home/zhanmusi/Dropbox/Birkbeck/' +
        'Advanced Computing Technologies MSc/' +
        'Project/Code/Initial Testing/' + modelFile
    )
    global WAV_LENGTH
    global GENERATOR
    global OUTPUT_DIR
    WAV_LENGTH = wave_length
    GENERATOR = model
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


def _convert_to_int(samples):
    """ Converts floats to integers """
    ints = samples * ABS_INT16
    ints = np.clip(ints, -ABS_INT16, ABS_INT16)
    ints = ints.astype(ctypes.c_int16)
    return ints


if __name__ == "__main__":
    main()
