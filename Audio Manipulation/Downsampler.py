import argparse as ag
import librosa as lb
import numpy as np
import os
import soundfile as sf


# Constants
IN_PATH = '/home/zhanmusi/Documents/Data/Speech Commands Dataset/'
OUT_PATH = '/home/zhanmusi/Documents/Data/Speech Commands Dataset Downsampled/'
WAV_LENGTH = 16384
SAMP_RATE = None


def main(args):
    """ Runs the code """
    global SAMP_RATE
    SAMP_RATE = int(WAV_LENGTH / args.divider[0])
    _loopFolders(args.words)
    return


def _loopFolders(folders):
    """ Loops through all folders found at the IN_PATH """
    for folder in folders:
        path = IN_PATH + folder + '/'
        _loopFiles(path, folder)
    return


def _loopFiles(path, folder):
    """ Loop through all files found within the folder  """
    allFiles = lb.util.find_files(path, ext='wav')
    for eachFile in allFiles:
        filePath, fileName = os.path.split(eachFile)
        resampled = _resampleFile(eachFile)
        _saveFile(folder, resampled, fileName)
    return


def _resampleFile(wav):
    """ Resample the wav file passed to the function """
    series, sampRate = lb.core.load(wav, sr=None)
    newSeries = _standardizeLength(series)
    resampled = lb.core.resample(
        y=newSeries,
        orig_sr=WAV_LENGTH,
        target_sr=SAMP_RATE)
    return resampled


def _standardizeLength(series):
    """ Standardizes the length of the wav before resampling """
    if len(series) < WAV_LENGTH:
        series = np.append(series, np.zeros(WAV_LENGTH - len(series)))
    elif len(series) > WAV_LENGTH:
        series = series[:WAV_LENGTH]
    return series


def _saveFile(folder, resampled, fileName):
    """ Save the file """
    path = OUT_PATH + str(SAMP_RATE) + '/' + folder + '/'
    if not os.path.exists(path):
        os.makedirs(path)
    sf.write(
        file=path + fileName,
        data=resampled,
        samplerate=SAMP_RATE,
        subtype='PCM_16')
    return


if __name__ == "__main__":
    parser = ag.ArgumentParser(description="Pass a downsample divider:")
    parser.add_argument(
        dest='divider',
        nargs=1,
        type=int,
        choices=[2, 4, 8, 16],
        help="A divider for downsampling."
    )
    parser.add_argument(
        dest='words',
        nargs='*',
        type=str,
        default=['zero', 'one'],
        help="Names of words (and folders) to be resampled."
    )
    main(parser.parse_args())
