function [ Ytrain ] = getTrainingSamples( dirMapDB )
home =cd;
trainList = dirMapDB;
i = 1;
Ytrain = struct([]);
upperf  = 30e3; % Chosen based on SERDP MR-1665-FR Final Report
f_s     = 200e3;%
eps     = 1; %Experimentally determined threholding value for grabbing important aspects
aper = 186; % half degree resolution
sigN=301;

    for obj = 1:size(trainList,2)
      DD=[];
      for e = 1:size(trainList,1);
        for tag = trainList(e,obj,:,:);
            cd(char(tag)); % navigate to the directory in dirMapDB=trainList
            x = what;
            x = x.mat; %grab the string of the .mat in directory
            ob = open(char(x));
            %AC = ob.acp'; % Opening AC in directory
            pings = ob.pings';
            cd(home);
            %DD = [DD,formatACLsas(AC,301)]; % This script outputs AC's w/ N=301
            DD = [DD, extractAC(pings,eps,sigN,upperf, f_s,aper)];
            i = i +1;
        end
      end
      Ytrain(obj).D = DD;
    end

end

