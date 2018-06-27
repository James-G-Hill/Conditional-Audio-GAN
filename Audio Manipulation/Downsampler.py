import argparse as ag
import librosa as lb
import os
import soundfile as sf


# The paths for loading & saving the .wav files
inPath = '/home/zhanmusi/Documents/Data/Speech Commands Dataset/'
outPath = '/home/zhanmusi/Documents/Data/Speech Commands Dataset Downsampled '
sampleRate = 16000

# List of folders the word recordings should be extracted from
folders = ['zero', 'one']


def main(divideBy):
    newSampleRate = sampleRate / divideBy
    fullOutPath = outPath + newSampleRate + '/'
    loopFolders(fullOutPath)
    return


def loopFolders(fullOutPath):
    for folder in fullOutPath:
        loopFiles(folder)
    return


def loopFiles(folder):
    allFiles = lb.util.find_files(inPath + folder + '/', ext='wav')
    for eachFile in allFiles:
        filePath, fileName = os.path.split(eachFile)
        resampled = resampleFile(eachFile)
        saveFile(folder, resampled, fileName)
    return


def resampleFile(wav):
    return


def saveFile(folder, resampled, fileName):
    sf.write(
        file=outPath + folder + '/' + fileName,
        data=resampled,
        samplerate=newSampleRate,
        subtype='PCM_16')
    return


if __name__ == "__main__":
    parser = ag.ArgumentParser(description="Pass the downsample divider:")
    parser.add_argument(
        'divider',
        type=int,
        required=True,
        help="A divider for downsampling")
    parser.parse_args()
