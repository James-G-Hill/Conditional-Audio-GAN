import argparse as ag
import librosa as lb
import os
import soundfile as sf


# The paths for loading & saving the .wav files
inPath = '/home/zhanmusi/Documents/Data/Speech Commands Dataset/'
outPath = '/home/zhanmusi/Documents/Data/Speech Commands Dataset Downsampled '
sampleRate = 16000
newSampleRate = 0

# List of folders the word recordings should be extracted from
folders = ['zero', 'one']


def main(divideBy):
    """ Runs the code """
    global newSampleRate
    newSampleRate = sampleRate / divideBy
    loopFolders()
    return


def loopFolders():
    """ Loops through all folders found at the inPath """
    for folder in folders:
        path = inPath + folder + '/'
        loopFiles(path)
    return


def loopFiles(folder):
    """ Loop through all files found within the folder  """
    allFiles = lb.util.find_files(folder, ext='wav')
    for eachFile in allFiles:
        filePath, fileName = os.path.split(eachFile)
        resampled = resampleFile(eachFile)
        saveFile(folder, resampled, fileName)
    return


def resampleFile(wav):
    """ Resample the wav file passed to the function """
    series, sampRate = lb.core.load(wav, sr=None)
    resampled = lb.core.resample(
        y=series,
        orig_sr=sampRate,
        target_sr=newSampleRate)
    return resampled


def saveFile(folder, resampled, fileName):
    """ Save the file """
    sf.write(
        file=outPath + folder + '/' + fileName,
        data=resampled,
        samplerate=newSampleRate,
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
