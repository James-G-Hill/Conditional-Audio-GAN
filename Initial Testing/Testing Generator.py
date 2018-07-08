import importlib.machinery as im
import random
import soundfile as sf
import types

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
    global DISCRIMINATOR
    DISCRIMINATOR = model
    global OUTPUT_DIR
    OUTPUT_DIR = '/home/zhanmusi/Documents/Data/Generated Samples/' \
                 + str(WAV_LENGTH)
    return


def generatorSamples():
    """ Generates samples """
    z_samples = _createZs()
    samples = GENERATOR.generate(z_samples)
    _writeSamples(samples)
    return


def _createZs():
    """ Creates randomly sampled z inputs for generator """
    lst = []
    for i in range(1, BATCH_SIZE):
        sample = random.sample(range(-16000, 16000), WAV_LENGTH)
        lst.append(sample)
    return lst


def _writeSamples(samples):
    for i in range(1, len(samples)):
        sf.write(
            file=OUTPUT_DIR + i + '.wav',
            data=samples(i, ),
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

    return
