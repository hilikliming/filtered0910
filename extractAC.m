function [ usableAsps ] = extractAC( filtered_run,eps,sigN,upperf,f_s,stopbins )
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
run = double(filtered_run);
len = size(run,2);

ctr = findArcCtr(run,0.08*mean(mean(abs(run).^2)));

%% Begin Unpacking 
AC  = abs(fft(run,[],1));

% Windowing Useable spectral power
binsize = f_s/size(AC,1);
AC      = AC(1:ceil(upperf/binsize),:);

% Setting up smaller data structures for decimated form
aAC = zeros(size(AC,1),stopbins);
% Decimate along aspect if aperture is smaller than number of samples we
% have...
AC = AC(:,sum(AC,1)>1);

if(stopbins<size(AC,2))
    for i = 1:size(AC,1)
    aAC(i,:) = resample(double(AC(i,:)),stopbins,size(AC,2));
    end
else
    aAC = AC;
end

wid = 60;
% Select enhanced aspects in the in center (the 'norm' ones all have this)
if(strcmp(eps,'ctr'))
    ctr = floor(ctr*stopbins/size(run,2));
    aAC = aAC(:,ctr-wid:ctr+wid-1);
else
    if(strcmp(eps,'pwr'))% Selecting top aspects by ordering power
        [pwrss, order] = sort(sum(abs(aAC).^2,1));
        order = fliplr(order);
        order = order(1:floor(length(order)*0.750));
        aAC = aAC(:,order);
    else
        % Selecting top aspects by relative power
        aAC = aAC(:,sum(abs(aAC).^2,1)>eps);%*max(sum(abs(aAC).^2,1)));
    end
end
% Simulating Sparser stopping by taking every tenth aspect with a wobble of
% +-5 stops
aAC = reSort(aAC,12,4);
% For each aspect, decimate frequency bins to desired signal length
rAC = zeros(sigN,size(aAC,2));

for i = 1:size(aAC,2)
rAC(:,i) = abs(resample(aAC(:,i),sigN,size(aAC,1)))';
end

usableAsps = normc(rAC);%20*log10(abs(normc(rAC)));%
end

function [ctrStop]=findArcCtr(run,thresh)
    ctrStop = size(run,2)/2;
    ctrCol = size(run,1)-10;
    for stop = 1:size(run,2)
        firstReturn = find(abs(run(:,stop)).^2 > thresh,1,'first');
        if(~isempty(firstReturn)&& firstReturn<=ctrCol)
               ctrStop = stop;
               ctrCol  = firstReturn;

        end
    end
end

% Simulat
function [AC]=reSort(rAC,jump,nbhdsz)
AC = [];
wobble = 0;
i = 0;
    while size(rAC,2)>= (mod(jump*(i)+wobble,size(rAC,2)-1)+1)
        AC = [AC, rAC(:,(mod(jump*(i)+wobble,size(rAC,2)-1)+1))];
        rAC(:,(mod(jump*(i)+wobble,size(rAC,2)-1)+1)) = [];
        wobble = round((rand(1)-0.5)*nbhdsz);
        i = i +1;
    end
end




