clear all,
X2 = zeros([1,256])
X2(0+1) = 640
X2(16+1) = 256*exp(j*pi/3)
X2(240+1) = 256*exp(-j*pi/3)
x2 = ifft(X2)

