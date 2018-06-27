import argparse as ag
import librosa as lb
import numpy as np
import os
import soundfile as sf


# Constants
IN_PATH = '/home/zhanmusi/Documents/Data/Speech Commands Dataset/'
OUT_PATH = '/home/zhanmusi/Documents/Data/Speech Commands Dataset Downsampled '
WAV_LENGTH = 16384
SAMP_RATE = 8192

# List of folders the word recordings should be extracted from
folders = ['zero', 'one']


def main(divideBy):
    """ Runs the code """
    global SAMP_RATE
    SAMP_RATE = WAV_LENGTH / divideBy
    loopFolders()
    return


def loopFolders():
    """ Loops through all folders found at the IN_PATH """
    for folder in folders:
        path = IN_PATH + folder + '/'
        loopFiles(path)
    return


def loopFiles(folder):
    """ Loop through all files found within the folder  """
    allFiles = lb.util.find_files(folder, ext='wav')
    for eachFile in allFiles:
        filePath, fileName = os.path.split(eachFile)
        resampled = resampleFile(standardFile)
        saveFile(folder, resampled, fileName)
    return


def resampleFile(wav):
    """ Resample the wav file passed to the function """
    series, sampRate = lb.core.load(wav, sr=None)
    newSeries = standardizeLength(series)
    resampled = lb.core.resample(
        y=newSeries,
        orig_sr=sampRate,
        target_sr=SAMP_RATE)
    return resampled


def standardizeLength(series):
    """ Standardizes the length of the wav before resampling """
    if len(series) < WAV_LENGTH:
        series = np.append(series, np.zeros(WAV_LENGTH - len(series)))
    elif len(series) > WAV_LENGTH:
        series = series[:WAV_LENGTH]
    return series


def saveFile(folder, resampled, fileName):
    """ Save the file """
    sf.write(
        file=OUT_PATH + folder + '/' + fileName,
        data=resampled,
        samplerate=SAMP_RATE,
        subtype='PCM_16')
    return


if __name__ == "__main__":
    parser = ag.ArgumentParser(description="Pass a downsample divider:")
    parser.add_argument(
        'divider',
        type=int,
        required=True,
        help="A divider for downsampling")
    main(parser.parse_args())
