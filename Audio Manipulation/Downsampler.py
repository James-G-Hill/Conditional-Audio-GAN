import librosa as lb
import os
import soundfile as sf


# The paths for loading & saving the .wav files
inPath = '/home/zhanmusi/Documents/Data/Speech Commands Dataset/'
outPath = '/home/zhanmusi/Documents/Data/Speech Commands Dataset Downsampled/'

# List of folders the word recordings should be extracted from
folders = ['zero', 'one']

# divideBy can be changed to determine the downsample rate
divideBy = 4
sampleRate = 16000
newSampleRate = sampleRate / divideBy


def main():
    loopFolders()
    return


def loopFolders():
    for folder in folders:
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
    main()
