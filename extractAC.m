function [ usableAsps ] = extractAC( filtered_run,eps,sigN,upperf,f_s,aper )
%% Input
% dir - directory to find Target Data
% filtered_run - matrix containing filtered run from Pond or TREX to be
% converted to a target strength AC
% eps - threshold for useful aspects for classification
% dim - array with the dimensions desired for the output (aspect res, freq
% res)
%% Output
% useableAsps - AC matrix for aspects with meaningful power (absolute sum
% greater than eps
run = filtered_run;
len = size(run,2);

%% Begin Unpacking 
AC  = abs(fft(run,[],2))';

% Windowing Useable spectral power
binsize = f_s/size(AC,1);
AC      = AC(1:ceil(upperf/binsize),:);

% Setting up smaller data structures for decimated form
aAC = zeros(size(AC,1),aper);
% Decimate along aspect
for i = 1:size(AC,1)
aAC(i,:) = resample(double(AC(i,:)),aper,size(AC,2));
end
% Select enhanced aspects in the in center (the 'norm' ones all have this)
aAC = aAC(:,sum(abs(aAC),1)>eps);

% For each aspect, decimate frequency bins to desired signal length
rAC = zeros(sigN,size(aAC,2));
for i = 1:size(aAC,2)
rAC(:,i) = abs(resample(aAC(:,i),sigN,size(aAC,1)))';
end

usableAsps = normc(rAC);
end