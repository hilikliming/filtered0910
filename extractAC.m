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

%% Begin Unpacking and setting up smaller data structures for decimated form
AC = abs(fft(run,[],2))';
aAC = zeros(size(AC,1),aper);
AC = AC(:,sum(abs(AC),1)>1);
% Decimate along aspect
for i = 1:size(AC,1)
aAC(i,:) = resample(AC(i,:),aper,size(AC,2));
end

% Removing useless aspects
aAC = aAC(:,sum(abs(aAC),1)>eps);
%AC = AC(:,1:602);
binsize = f_s/size(AC,1);
% Windowing Useable spectral power
aAC = aAC(1:ceil(upperf/binsize),:);

rAC = zeros(sigN,size(aAC,2));
% For each aspect, decimate frequency bins
for i = 1:size(aAC,2)
rAC(:,i) = abs(resample(aAC(:,i),sigN,size(aAC,1)))';
end

% for asp = 1:size(rAC,2)
%     %rAC(:,asp) = 20*log10(abs(rAC(:,asp))/norm(rAC(:,asp))); % Abs(.) added to flip negative ripple effects of resample
%     rAC(:,asp) = rAC(:,asp)/norm(rAC(:,asp));
% end

usableAsps = normc(rAC);
end