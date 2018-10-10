import csv
import librosa as lb
import numpy as np
import os
import pylab as pl
import soundfile as sf


def soundInfo(fileName, series, sampRate):
    """ Print out information about a sound """
    print('File name              : ' + fileName)
    print('Data type              : ' + str(series.dtype))
    print('Sample points          : ' + str(series.shape[0]))
    print('Sample rate            : ' + str(sampRate))
    print('Duration               : ' +
          str(lb.core.get_duration(series, sampRate)))
    print('\n')
    return


def plotTone(series, sampRate):
    """ Plot a tone to a graph """
    # Convert sound array to floating point between -1 to 1
    snd = series / (2. ** 15)
    timeArray = pl.arange(0, snd.shape[0])
    timeArray = timeArray / sampRate
    timeArray = timeArray * 1000  # scales to milliseconds
    # Plot the tone graph
    np.plot(timeArray, snd, color='k')
    np.ylabel('Amplitude')
    np.xlabel('Time (ms)')
    np.plt.show()
    return


def plotFrequency(series, sampRate):
    """ Plot a frequency to a graph """
    s = series / (2. ** 15)
    p, freqArray = _fastFourier(s, sampRate)
    np.plot(freqArray / 1000, 10 * pl.log10(p), color='k')
    np.xlabel('Frequency (kHz)')
    np.ylabel('Power (dB)')
    np.plt.show()
    return


def compareRoots(series, sampRate):
    """ Compare the roots to see that they are equal """
    s = series / (2. ** 15)
    p, freqArray = _fastFourier(s, sampRate)
    rms_val = pl.sqrt(pl.mean(s ** 2))
    print('Root mean square value : ' + str(rms_val))
    print('Square root            : ' + str(pl.sqrt(sum(p))))
    return


def _fastFourier(series, sampRate):
    """ Perform a Fast Fourier Transform """
    n = len(series)
    p = sf.fft(series)  # Fast fourier transform
    uUniquePts = int(pl.ceil((n+1) / 2.0))
    p = p[0:uUniquePts]
    p = abs(p)
    p = p / float(n)
    p = p ** 2
    if n % 2 > 0:
        p[1:len(p)] = p[1:len(p)] * 2
    else:
        p[1:len(p) - 1] = p[1:len(p) - 1] * 2
    freqArray = np.arange(0, uUniquePts, 1.0) * (sampRate / n)
    return (p, freqArray)


def showWaveplot(series, sampRate):
    """ Plots the wave of the sound """
    pl.display.waveplot(series, sampRate)
    pl.plt.show()
    return


def copyDownSampleFolder(inFilePath, folderName, outFilePath, newSampRate):
    """ Copies folder of samples to a new sample rate """
    files = lb.util.find_files(inFilePath + folderName + '/', ext='wav')
    for file in files:
        path, name = os.path.split(file)
        series, sampRate = lb.core.load(file, sr=None)
        reSampled = lb.core.resample(
            y=series,
            orig_sr=sampRate,
            target_sr=newSampRate)
        sf.write(
            file=outFilePath + folderName + '/' + name,
            data=reSampled,
            samplerate=newSampRate,
            subtype='PCM_16')
    return


def createTrainingDataCSV(folders, outFilePath):
    """ Creates a training file as .csv """
    trainData = []
    trainLabel = []
    for folder in folders:
        files = lb.util.find_files(outFilePath + folder + '/', ext='wav')
        for file in files:
            path, name = os.path.split(file)
            series, sampRate = lb.core.load(file, sr=None)
            trainData.append(series)
            trainLabel.append(folder)
    with open(outFilePath + 'binaryTrainingData', 'w') as dataFile:
        wr = csv.writer(dataFile, delimiter=',', quoting=csv.QUOTE_ALL)
        wr.writerow(trainData)
    with open(outFilePath + 'binaryTrainingLabels', 'w') as labelFile:
        wr = csv.writer(labelFile, delimiter=',', quoting=csv.QUOTE_ALL)
        wr.writerow(trainLabel)
    return


inFilePath = '/home/zhanmusi/Documents/Data/Speech Commands Dataset/'
inFileFolder = 'zero'
inFileName = '0ab3b47d_nohash_0.wav'
inFullPath = inFilePath + inFileFolder + '/' + inFileName

outFilePath = \
    '/home/zhanmusi/Documents/Data/Speech Commands Dataset Downsampled/'
outFileFolder = 'zero'
outFileName = inFileName
outFullPath = outFilePath + outFileFolder + '/' + outFileName

series, sampRate = lb.core.load(inFullPath, sr=None)
reSampled = lb.core.resample(series, sampRate, int(sampRate / 2))

binaryDatasetList = ['zero', 'one']

# exec(open('Basic_Sound_Processing.py').read())
