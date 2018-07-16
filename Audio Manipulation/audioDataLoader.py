import hashlib as hl
import librosa as lb
import os
import re


MAX_NUM_WAVS_PER_CLASS = 2**27 - 1

LOOKUP = None

TEST_DATA = None
TEST_LABELS = None

TRAIN_DATA = None
TRAIN_LABELS = None

VALID_DATA = None
VALID_LABELS = None

TEST_PER = 20
VALID_PER = 5


def prepareData(inFilePath, folderNames):
    """ Loads the audio data & labels into list form """
    _resetGlobalVariables()
    label = 0
    for folder in folderNames:
        files = lb.util.find_files(inFilePath + folder + '/', ext='wav')
        _appendInfo(files, label)
        LOOKUP.append((label, folder))
        label = label + 1
    return


def loadTrainData():
    """ Returns the training data only """
    return TRAIN_DATA, TRAIN_LABELS


def loadTestData():
    """ Returns the evaluation data only """
    return TEST_DATA, TEST_LABELS


def loadValidData():
    """ Returns the validation data only """
    return VALID_DATA, VALID_LABELS


def getLookup():
    """ Returns the lookup data for the categories """
    return LOOKUP


def _resetGlobalVariables():
    """ Resets the variables if the module has already been used """
    global TEST_DATA
    global TEST_LABELS
    global TRAIN_DATA
    global TRAIN_LABELS
    global VALID_DATA
    global VALID_LABELS
    global LOOKUP
    TEST_DATA = []
    TEST_LABELS = []
    TRAIN_DATA = []
    TRAIN_LABELS = []
    VALID_DATA = []
    VALID_LABELS = []
    LOOKUP = []
    return


def _appendInfo(files, label):
    """ Appends file data series & label to the lists """
    for eachFile in files:
        series, sampRate = lb.core.load(eachFile, sr=None)
        path, fileName = os.path.split(eachFile)
        hashPercent = _getPercHash(fileName)
        if hashPercent < VALID_PER:
            VALID_DATA.append(series)
            VALID_LABELS.append(label)
        elif hashPercent < (TEST_PER + VALID_PER):
            TEST_DATA.append(series)
            TEST_LABELS.append(label)
        else:
            TRAIN_DATA.append(series)
            TRAIN_LABELS.append(label)
    return


def _getPercHash(name):
    """ Calculates & returns the percentage from the hash """
    hash_name = re.sub('_nohash_.*$', '', name)
    hash_name_encode = hash_name.encode('utf-8')
    hash_name_hashed = hl.sha1(hash_name_encode).hexdigest()
    percent_hash = (
        (int(hash_name_hashed, 16) % (MAX_NUM_WAVS_PER_CLASS + 1))
        * (100.0 / MAX_NUM_WAVS_PER_CLASS))
    return percent_hash
