function [ Ytrain ] = getTrainingSamples( dirMapDB )
home =cd;
trainList = dirMapDB;
i = 1;
Ytrain = struct([]);

  for obj = 1:size(trainList,2)
      DD=[];
      for e = 1:size(trainList,1);
        for tag = trainList(e,obj,:,:);
            cd(char(tag));
            x = what;
            x = x.mat;
            ob = open(char(x));
            AC = ob.acp'; % Opening AC in directory
            cd(home);
            DD = [DD,formatACLsas(AC,301)]; % This script outputs target strength
            i = i +1;
        end
      end
      Ytrain(obj).D = DD;
   end


end

