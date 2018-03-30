from librosa import* 
from pylab import*


def soundInfo(fileName, series, sampRate):
    """ Print out information about a sound """
    print('File name              : ' + fileName)
    print('Data type              : ' + str(series.dtype))
    print('Sample points          : ' + str(series.shape[0]))
    print('Sample frequency       : ' + str(sampRate))
    print('Sample rate            : ' + str(series.shape[0] / sampRate))
    print('\n')
    return


def plotTone(s, sf):
    """ Plot a tone to a graph """
    # Convert sound array to floating point between -1 to 1
    snd = s / (2.**15)
    timeArray = arange(0, snd.shape[0])
    timeArray = timeArray / sf
    timeArray = timeArray * 1000 # scales to milliseconds
    # Plot the tone graph
    plot(timeArray, snd, color='k')
    ylabel('Amplitude')
    xlabel('Time (ms)')
    plt.show()
    return


def plotFrequency(s, sf):
    """ Plot a frequency to a graph """
    s = s / (2.**15)
    p, freqArray = _fastFourier(s, sf)
    plot(freqArray/1000, 10*log10(p), color='k')
    xlabel('Frequency (kHz)')
    ylabel('Power (dB)')
    plt.show()
    return


def compareRoots(s, sf):
    """ Compare the roots to see that they are equal """
    s = s / (2.**15)
    p, freqArray = _fastFourier(s, sf)
    rms_val = sqrt(mean(s**2))
    print('Root mean square value : ' + str(rms_val))
    print('Square root            : ' + str(sqrt(sum(p))))
    return


def _fastFourier(s, sf):
    """ Perform a Fast Fourier Transform """
    n = len(s)
    p = fft(s) # Fast fourier transform
    uUniquePts = int(ceil((n+1)/2.0))
    p = p[0:uUniquePts]
    p = abs(p)
    p = p / float(n)
    p = p**2
    if n % 2 > 0:
        p[1:len(p)] = p[1:len(p)] * 2
    else:
        p[1:len(p) - 1] = p[1:len(p) - 1] * 2
        
    freqArray = arange(0, uUniquePts, 1.0) * (sf / n);
    return (p, freqArray)


filePath         = '/home/zhanmusi/Documents/Data/Speech Commands Dataset/'
fileFolder       = 'zero/'
fileName         = '0ab3b47d_nohash_0.wav'
fullPath         = filePath + fileFolder + fileName
series, sampRate = core.load(fullPath, sr=None)


# exec(open('Basic_Sound_Processing.py').read())
