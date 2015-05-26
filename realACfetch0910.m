function [ Y, t_Y, Dclutter ] = realACfetch0910( realTarg )
home     = cd;
targetID = 1;
Y   = []; 
t_Y = [];
upperF  = 31e3; % Chosen based on SERDP MR-1665-FR Final Report
f_s     = 100e3;%
sig_N   = 310;  % (0-31 kHz)
eps      = 30; %Experimentally determined threholding value for grabbing important aspects
Dclutter = [];
aper = 750;
for tag = realTarg
    cd(['C:\Users\halljj2\Desktop\WMSC-CODE\UW Pond\TARGET_DATA\',char(tag),'\PROUD_10m']);
    here = cd; % Where the object run .mat files are located
    x = what; x = x.mat;
    % Various rotations of orientation are captured in each run
    for run = 1:length(x)
        if(strfind(char(x(run)),'_norm')) % if it's well conditioned type
            ob=open(char(x(run))); % char() just converts the cell to string
            cd(home);
            %aper= size(ob.new_data,1);%TEMPORARILY USING ALL PING RESOLUTION
            AC = extractAC(ob.new_data,eps,sig_N,upperF,f_s,aper);
            if(strfind(char(x(run)),'ROCK'))
                Dclutter = [Dclutter,AC];
            else
                % Extracting AC template from filtered run
                Y   = [Y, AC];
                t_Y = [t_Y', targetID*ones(size(AC,2),1)']';
            end
            cd(here);
        end
    end
    targetID = targetID + 1;  
end
cd(home);
end
