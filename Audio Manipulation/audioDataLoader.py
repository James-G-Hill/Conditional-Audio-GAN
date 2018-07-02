import librosa as lb


DATA = None
LABELS = None
LOOKUP = None

TEST_PER = 20
EVAL_PER = 5


def loadData(inFilePath, folderNames):
    """ Loads the audio data & labels into list form & returns """
    _resetGlobalVariables()
    label = 0
    for folder in folderNames:
        files = lb.util.find_files(inFilePath + folder + '/', ext='wav')
        _appendInfo(files, label)
        LOOKUP.append((label, folder))
        label = label + 1
    return DATA, LABELS, LOOKUP


def _resetGlobalVariables():
    """ Resets the variables if the module has already been used """
    global DATA
    global LABELS
    global LOOKUP
    DATA = []
    LABELS = []
    LOOKUP = []
    return


def _appendInfo(files, label):
    """ Appends file data series and label to the lists """
    for eachFile in files:
        series, sampRate = lb.core.load(eachFile, sr=None)
        DATA.append(series)
        LABELS.append(label)
    return


def _splitForTesting():
    """ Splits the datasets into training, testing & evaluation sets """
    return
