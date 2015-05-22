% Input obj is a cell of strings with object names. range rho_w, rho_s are vectors of ranges water density
% etc.

function [ dirMap ] = buildDirsLsas(dir,exes,ins,envs,obj, ranges, aspects,runlen )
% Returned from function, contains strings of each of the sub directories
% containing halfspace.in and acolor.in...
home = cd;
cd(dir);
[nenvs, ~]=size(envs);
dirMap = cell(nenvs,length(obj),length(ranges),length(aspects));
% Creating Numbered Environment Directories

for e = 1:nenvs
    objEnvDir = ['ENV_' num2str(e)];
    mkdir(objEnvDir);
    cd(objEnvDir);
    % Creating labeled Object Sub-Directories
    for o = 1:length(obj)
        objDir = char(obj(o));
        mkdir(objDir);
        cd(objDir)
        % Creating numbered Range Sub-Directories
        for r = 1:length(ranges)
            objRangeDir = char([num2str(ranges(r)) '_m']);
            mkdir(objRangeDir);
            cd(objRangeDir);
            copyfile(exes,cd);
            % Creating numbered Aspect Sub-Directories and copying over
            % dummy .in's from the template directory... These will later
            % be altered for specific environment information
            for a = 1:length(aspects)
                objAspDir = char([num2str(aspects(a)) '_deg']);
                mkdir(objAspDir);
                % Copying the Dummy .in Files to the sub folder
                % corresponding the the appropriate
                % Environment,object,range,aspect
                copyfile(ins,objAspDir);
                cd(objAspDir);
                dirMap(e,o,r,a) = cellstr(cd);
                run_el = createLsasRun(runlen(2),envs(e,3),runlen(1));
                hndl = ['lsas.' num2str(runlen(1)) 'm.dat'];
                dlmwrite(hndl,run_el)
                %save(hndl, 'run_el');                
                cd('..\');
            end
            cd('..\')
        end
        cd('..\');
    end
    cd('..\');
end
cd(home);
end

