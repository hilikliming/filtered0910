function [ Ytrain ] = getTrainingSamples( dirMapDB ,stops)
home =cd;
trainList = dirMapDB;
i = 1;
Ytrain = struct([]);
upperf  = 31e3; % Chosen based on SERDP MR-1665-FR Final Report
f_s     = 100e3;%
eps     = 1e-2; %Experimentally determined threholding value for grabbing important aspects
sigN    = 310;

    for obj = 1:size(trainList,2)
      DD=[];
      for e = 1:size(trainList,1);
        for tag = trainList(e,obj,:,:);
            cd(char(tag)); % navigate to the directory in dirMapDB=trainList
            x = what;
            x = x.mat; %grab the string of the .mat in directory
            ob = open(char(x));
            %AC = ob.acp'; % Opening AC in directory
            pings = ob.pings;
            
            cd(home);
            xmit = ob.signal;
            %xmit = xmit(:,2);%[xmit(:,2)' zeros(1,size(pings,1)-length(xmit))]';%.*xmit(:,1);
            %pings = PulseCompress(pings,xmit,ob.sample_rate,200);
%             for asp = 1:size(pings,2)
%                 strip = conv(fliplr(xmit)',pings(:,asp));
%                 pings(:,asp) = strip(1:size(pings,1));%PulseCompress(pings,xmit,ob.sample_rate,0.5e3);
%             end
            %DD = [DD,formatACLsas(AC,301)]; % This script outputs AC's w/ N=301
            DD = [DD, extractAC(pings,'pwr',sigN,upperf,f_s,stops)];
            i = i +1;
        end
      end
      Ytrain(obj).D = DD;
    end

end

