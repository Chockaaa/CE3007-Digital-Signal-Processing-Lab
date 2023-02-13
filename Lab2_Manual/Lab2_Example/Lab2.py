# -*- coding: utf-8 -*-
"""
Created on Fri Feb 10 14:21:41 2023

@author: Kai Jun
"""
import numpy as np;
import matplotlib.pyplot as plt;
import scipy.io.wavfile  as wavfile;
import winsound;
import os;
from scipy import signal;

# The input is a float array (should have dynamic value from -1.00 to +1.00
def fnNormalizeFloatTo16Bit(yFloat):
    y_16bit = [int(s*32767) for s in yFloat]
    return(np.array(y_16bit, dtype='int16'))

# The input is a float array (should have dynamic value from -1.00 to +1.00
def fnNormalize16BitToFloat(y_16bit):
    yFloat = [float(s/32767.0) for s in y_16bit]
    return(np.array(yFloat, dtype='float'))

def simpleCos():
    cycles = 5;
    A = 1;
    w = 0.1 * np.pi;
    Fs = 10000;
    sTime = 0;
    eTime = cycles * (2*np.pi / w);
    n = np.arange(sTime, eTime, 1/Fs);
    x = A * np.cos(w * n);
    plt.figure();
    plt.plot(n,x);
    pass;
    
def myPlotStem(impulseH, title, color = 'b'):
    plt.figure();
    plt.stem(impulseH, color);
    plt.title(title);
    plt.show();
    pass;
    
def myPlot(y, title, color = 'b'):
    plt.figure();
    plt.plot(y, color);
    plt.title(title);
    plt.show(); 
    pass;

def getSoundFile(voiceFile, sound):
    if sound:
        winsound.PlaySound(voiceFile, winsound.SND_FILENAME);
    
    [Fs, sampleX_16bit] = wavfile.read(voiceFile);
    
    return [Fs, sampleX_16bit];
    

def myConvolve(x,y,mode='full'):
    result = [];
    a, v = x,y;
    if (len(v) > len(a)):    #swap the longer array to a
        a, v = v, a

    if mode == 'full':      #pad to a

        pad = np.array([0 for _ in range(len(v)-1)]);
        b = np.copy(a);
        a = np.concatenate((pad,b,pad));
    for j in range(len(a)-len(v)+1):
        result.append((np.sum(np.matmul(v[::-1],a[j:j+len(v)]))));
    return np.array(result);

def playSoundFromWav(yFloat, Fs = 16000, filename = "t1_16bit.wav"):
    y_16bit = fnNormalizeFloatTo16Bit(yFloat);
    wavfile.write(filename, Fs, y_16bit);
    winsound.PlaySound(filename, winsound.SND_FILENAME);
    os.remove(filename);
    pass;

def plotSpectro(y, Fs, title):
    
    [f, t, Sxx_clean] = signal.spectrogram(y, Fs, window=('blackmanharris'));
    plt.pcolormesh(t, f, 10*np.log10(Sxx_clean));
    plt.ylabel('Frequency [Hz]');
    plt.xlabel('Time [sec]');
    plt.title(title);
    plt.show();
    
    pass;

def Lab2_1():
    
    # simpleCos();
    
    #Parameters
    A = 1;
    w = 0.1 * np.pi;
    numSamples = 3;
    
    n = np.array([0,1,2]);
    
    print(n);
    
    x = A * np.cos(w * n);
    print(x);
    h = np.array([0.2,0.3,-0.5], dtype = "float");
    print(h);
    y = np.convolve(x,h);

    plt.figure();
    plt.stem(n[0:numSamples], x[0:numSamples], 'b--', label = "Input");
    plt.stem(n[0:numSamples],y[0:numSamples], 'r--', label = "Output");
    plt.legend();
    plt.show();
    # plt.stem(y);
    
    pass;
    
def Lab2_3():
    impulseH = np.zeros(8000);
    impulseH[1] = 1;
    impulseH[4000] = 0.5;
    impulseH[7900] = 0.3;
    soundFile = "helloWorld_16bit.wav";

    # myPlotStem(impulseH, "ImpulseH");

    [Fs, sampleX_16bit] = getSoundFile(soundFile);
    
    sampleX_float = fnNormalize16BitToFloat(sampleX_16bit);
    sampleX_float = np.multiply(3.0,sampleX_float)
    
    plt.figure()
    plt.plot(sampleX_float,'r')
    plt.ylabel('signal (float)')
    plt.xlabel('sample n')
    plt.show()
    
    y1 = myConvolve(sampleX_float, impulseH);
    myPlot(y1, "myConvolve");
    
# =============================================================================
#     y2 = np.convolve(sampleX_float, impulseH);
#     plotY(y2, "np.convovle");
# =============================================================================
    playSoundFromWav(y1);
    pass;

def Lab2_4():
    h1 = np.array([0.06523, 0.14936, 0.21529, 0.2402, 0.21529, 0.14936, 0.06523], dtype = "float");
    h2 = np.array([-0.06523, -0.14936, -0.21529, 0.7598, -0.21529, -0.14923, -0.06523], dtype = "float");
    
    myPlotStem(h1, "h1", "r");
    myPlotStem(h2, "h2");
    
    x = np.zeros(16);
    x[0] = 1;
    x[15] = -2;
    myPlotStem(x, "x");
    
    y1 = myConvolve(h1, x);
    y2 = myConvolve(h2, x);
    
    myPlot(y1, "y1");
    myPlot(y2, "y2");
    pass;
    
def Lab2_4c():
    Cycles = 2000;
    A = 0.1;
    freq1 = 700;
    freq2 = 3333;
    Fs = 16000;
    
    sTime = 0;
    eTime = Cycles * (1/(min(freq1,freq2)));
    t = np.arange(sTime, eTime, 1/Fs);
    
    
    h1 = np.array([0.06523, 0.14936, 0.21529, 0.2402, 0.21529, 0.14936, 0.06523], dtype = "float");
    h2 = np.array([-0.06523, -0.14936, -0.21529, 0.7598, -0.21529, -0.14923, -0.06523], dtype = "float");
    
    x1 = A * np.cos(2 * np.pi * freq1 * t);
    x2 =  A * np.cos(2 * np.pi * freq2 * t);
    x =  x1 + x2;
    
    plt.figure();
    plt.plot(t, x);
    plt.show();
    
    y1 = myConvolve(h1, x);
    y2 = myConvolve(h2, x);
    
    plotSpectro(y1, Fs, "Y1 Spectrogram");
    plotSpectro(y2, Fs, "Y2 Spectrogram");
    
    playSoundFromWav(y1);
    playSoundFromWav(y2);
    
    pass;

def Lab2_5():
    filename = "helloworld_noisy_16bit.wav";
    [Fs, x_16bit] = getSoundFile(filename, 0);
    
    x_float = fnNormalize16BitToFloat(x_16bit);
    x_float = np.multiply(3.0,x_float)
    
    b = np.array([1, -0.7653668, 0.99999], dtype = "float");
    a = np.array([1, -0.722744, 0.888622], dtype = "float");
    plotSpectro(x_float, Fs, "x");
    
    y = signal.lfilter(b, a, x_float);
    plotSpectro(y, Fs, "y");
    
    playSoundFromWav(y, Fs);
    
    myPlot(x_float, "x");
    myPlot(y, "y");
    
    pass;
    
def Lab2_5f():
    delta = np.zeros(10);
    delta[0] = 1;
    
    b = np.array([1, -0.7653668, 0.99999], dtype = "float");
    a = np.array([1, -0.722744, 0.888622], dtype = "float");
    
    y = signal.lfilter(b, a, delta);
    myPlotStem(y, "Impulse Response");
    

def main():
    # Lab2_1();
    # Lab2_3();
    # Lab2_4();
    # Lab2_4c();
    # Lab2_5();
    Lab2_5f();
    
    pass;
    
if __name__ == "__main__":
    main();