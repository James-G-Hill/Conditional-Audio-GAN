import librosa as lb


TRAIN_DATA = None
TRAIN_LABELS = None
LABEL_LOOKUP = None


def loadData(inFilePath, folderNames):
    """ Loads the audio data & labels into list form & returns """
    _resetGlobalVariables()
    label = 0
    for folder in folderNames:
        files = lb.util.find_files(inFilePath + folder + '/', ext='wav')
        _appendInfo(files, label)
        LABEL_LOOKUP.append((label, folder))
        label = label + 1
    return TRAIN_DATA, TRAIN_LABELS, LABEL_LOOKUP


def _resetGlobalVariables():
    """ Resets the variables if the module has already been used """
    global TRAIN_DATA
    global TRAIN_LABELS
    global LABEL_LOOKUP
    TRAIN_DATA = []
    TRAIN_LABELS = []
    LABEL_LOOKUP = []
    return


def _appendInfo(files, label):
    """ Appends file data series and label to the lists """
    for eachFile in files:
        series, sampRate = lb.core.load(eachFile, sr=None)
        TRAIN_DATA.append(series)
        TRAIN_LABELS.append(label)
    return
