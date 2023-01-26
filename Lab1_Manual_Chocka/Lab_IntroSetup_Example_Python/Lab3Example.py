# -*- coding: utf-8 -*-
"""
Created on Sat Dec  2 13:10:08 2017

@author: Chng Eng Siong
Date: Dec 2017, using python to solve DSP problems
purpose - to remove dependency on Matlab
"""

# using quiver to plot the vectors 'arrow' showing the direction it is pointing to
# we plot for k=1, you can change this variable to see different harmonics.

import numpy as np
import matplotlib.pyplot as plt
import scipy.fftpack as fftpack


# You should check your implementation of Lab3 Question 2
# with respect to scipy.fftpack!


# Lab 3 Question 3, example code
# showing how to use quiver to plot the arrow for
# the Fourier basis at index k
N = 32
k=1
W = np.zeros(shape=(N),dtype=complex)
for n in np.arange(0,N):
    W[n] = np.exp(-1j*(2*np.pi/N)*k*n)

W_angle = np.angle(W)
# the lenbth is 1, we are only interested in the angle of each phasor

plt.figure()
plt.title('Each row shows the k-th harmonic, from n=0..N-1 index')
Q = plt.quiver( np.cos(W_angle),np.sin(W_angle),  units='width')
titleStr = 'Fourier complex vectors N='+str(N)
plt.title(titleStr)
plt.ylabel('k-values')
plt.xlabel('n-values')
plt.grid()
plt.show()
#showing the phasor in degree (not radian)
#print(np.angle(W)*180/np.pi)


# Lab 3 Question 5, example code
# showing how to use fft to convolve
# notice I DID not take care of the zero padding
# you will need to to make it work!
x = [1,2,0,0,0,0,0]
h = [1,2,3,0,0,0,0]
y = np.convolve(x,h)
print(y)

# you should write your own fft and ifft routine!
X = fftpack.fft(x)
H = fftpack.fft(h)
Y = np.multiply(X,H)
tst_y = fftpack.ifft(Y)
tst_y_abs = np.absolute(tst_y)
print(tst_y_abs)
