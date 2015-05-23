function [ Y, t_Y, Dclutter ] = realACfetch0910( realTarg )
home     = cd;
targetID = 1;
Y   = []; 
t_Y = [];
upperF  = 31e3; % Chosen based on SERDP MR-1665-FR Final Report
f_s     = 100e3;%

eps      = 30; %Experimentally determined threholding value for grabbing important aspects
Dclutter = [];

for tag = realTarg
    cd(['C:\Users\halljj2\Desktop\WMSC-CODE\UW Pond\TARGET_DATA\',char(tag),'\PROUD_10m']);
    here = cd; % Where the object run .mat files are located
    x = what; x = x.mat;
    % Various rotations of orientation are captured in each run
    for run = 1:length(x)
        if(strfind(char(x(run)),'norm')) % if it's well conditioned type
            ob=open(char(x(run))); % char() just converts the cell to string
            cd(home);
            TS = extractAC(ob.new_data,eps,301,upperF,f_s,185);
            if(strfind(char(x(run)),'ROCK'))
                Dclutter = [Dclutter,TS];
            else
                % Extracting AC template from filtered run
                Y   = [Y, TS];
                t_Y = [t_Y', targetID*ones(size(TS,2),1)']';
            end
            cd(here);
        end
    end
    targetID = targetID + 1;  
end
cd(home);
end
