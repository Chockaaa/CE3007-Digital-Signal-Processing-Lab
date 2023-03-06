clear all
X1 = [0,1,2,3,0,0,0,0]
N1 = 8
x_fft1 = fft(X1,N1)/N1
k=(0:7)
figure(1);
subplot(211)
stem(k,abs(x_fft1*2*pi ))
subplot(212)
stem(k,angle(x_fft1 ))