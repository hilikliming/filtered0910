function [ Ytrain ] = getTrainingSamples( dirMapDB )
home =cd;
trainList = dirMapDB;
i = 1;
Ytrain = struct([]);

for obj = 1:size(trainList,2)
  DD=[];
  for e = 1:size(trainList,1);
    for tag = trainList(e,obj,:,:);
        cd(char(tag)); % navigate to the directory in dirMapDB=trainList
        x = what;
        x = x.mat; %grab the string of the .mat in directory
        ob = open(char(x));
        AC = ob.acp'; % Opening AC in directory
        cd(home);
        DD = [DD,formatACLsas(AC,301)]; % This script outputs AC's w/ N=301
        i = i +1;
    end
  end
  Ytrain(obj).D = DD;
end


end

