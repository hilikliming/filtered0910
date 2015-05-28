function y = PulseCompress(x,xmit,fs,f0)

%********************************************************
% Take Hilbert and Fourier transform of transmit signal *
%********************************************************

xmit = hilbert(xmit);
Xmit = fft(xmit,size(x,1));

%*******************************************************
% Match filter and complex modulate (to baseband) data *
%*******************************************************

n = (0:size(x,1)-1)';
omega = (2*pi)*(f0/fs);

y = zeros(size(x));
for i = 1:size(x,2)
    y(:,i) = exp(-1i*omega*n).*ifft(fft(hilbert(x(:,i))).*conj(Xmit));
end