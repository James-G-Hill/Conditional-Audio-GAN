from pylab import*
from scipy.io import wavfile

# Code to run:
# exec(open('Basic_Sound_Processing.py').read())

##
## Load a wav file
##

filePath = '/home/zhanmusi/Documents/Data/Speech Commands Dataset/zero/'
fileName = '0ab3b47d_nohash_0.wav'
sampFreq, snd = wavfile.read(filePath + fileName)

print('Data type              : ' + str(snd.dtype))
print('Sample points          : ' + str(snd.shape[0]))
# print('Channel count          : ' + str(snd.shape[1]))
print('Sample frequency       : ' + str(sampFreq))
print('Sample rate            : ' + str(snd.shape[0] / sampFreq))

##
##  Plotting the Tone
##

# Convert sound array to floating point between -1 to 1
snd = snd / (2.**15)

timeArray = arange(0, snd.shape[0])
timeArray = timeArray / sampFreq
timeArray = timeArray * 1000 # scales to milliseconds

# Plot the tone graph
plot(timeArray, snd, color='k')
ylabel('Amplitude')
xlabel('Time (ms)')
plt.show()

##
##  Plotting the frequency content
##

n = len(snd)
p = fft(snd) # Fast fourier transform
uUniquePts = int(ceil((n+1)/2.0))
p = p[0:uUniquePts]
p = abs(p)
p = p / float(n)
p = p**2

if n % 2 > 0:
    p[1:len(p)] = p[1:len(p)] * 2
else:
    p[1:len(p) - 1] = p[1:len(p) - 1] * 2

freqArray = arange(0, uUniquePts, 1.0) * (sampFreq / n);

# Plot the frequency graph
plot(freqArray/1000, 10*log10(p), color='k')
xlabel('Frequency (kHz)')
ylabel('Power (dB)')
plt.show()

# Compute the root mean square
rms_val = sqrt(mean(snd**2))
print('Root mean square value : ' + str(rms_val))
print('Square root            : ' + str(sqrt(sum(p))))
