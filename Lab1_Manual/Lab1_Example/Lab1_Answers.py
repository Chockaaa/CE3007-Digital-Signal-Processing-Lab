import numpy as np
import matplotlib.pyplot as plt
import scipy.io.wavfile  as wavfile
import winsound
# plotting 3D complex plane
from mpl_toolkits.mplot3d import Axes3D

def fnGenSampledSinusoid(A,Freq,Phi, Fs,sTime,eTime):
    # Showing off how to use numerical python library to create arange
    n = np.arange(sTime,eTime,1.0/Fs)
    y = A*np.cos(2 * np.pi * Freq * n + Phi)
    return [n,y]


# The input is a float array (should have dynamic value from -1.00 to +1.00
def fnNormalizeFloatTo16Bit(yFloat):
    y_16bit = [int(s*32767) for s in yFloat]
    return(np.array(y_16bit, dtype='int16'))

# The input is a float array (should have dynamic value from -1.00 to +1.00
def fnNormalize16BitToFloat(y_16bit):
    yFloat = [float(s/32767.0) for s in y_16bit]
    return(np.array(yFloat, dtype='float'))


def q3a_generateSampledSignal(F=1000):
    A=0.1; Phi = 0;Fs=16000; sTime=0; eTime = 1
    [n,yfloat] = fnGenSampledSinusoid(A, F, Phi, Fs, sTime, eTime)
    plt.figure(1)
    numSamples =72
    plt.plot(n[0:numSamples], yfloat[0:numSamples],'r--o');
    plt.xlabel('time in sec'); plt.ylabel('y[nT]')
    plt.title('sinusoid of signal (floating point)')
    plt.grid()
    plt.show()
    print('Above figure 1 shows sinusoid')