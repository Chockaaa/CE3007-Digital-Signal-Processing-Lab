clear all
%x3 = [1,2,3,4] 
%h[n] = [1,1] 
N3 = 6  %and 5 6 7
x3 = ifft(fft([1,2,3,4],N3).*fft([1,1],N3))