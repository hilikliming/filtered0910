%% ***********************************************%%
%  CSAS ACP  Database Generator                   %%
%  John Hall                                      %%
%  CSU (SERDP Project)                            %%
%%************************************************%%

function [dirMap] = generateDatabaseLsas(dir,envs,ranges, rots, obs, chirp, runlen)
    %% INPUTS:
    % envs = cell array length of no. of environments filled with c_w, c_s
    % pairs and interface elevation
    % ranges = vector of ranges included in DB
    % aspects = vector of aspect angles of object in DB
    % objs = vector of object numbers to include
    % Number of Objects and template signals for classifier
    
    %% OUTPUTS:
    % dirMap % cell object containing strings of all the subdirectories
    % containing the various ACP images
    home = cd;
    %% Predetermined Object Data
    N_objs = 10;
    
    objs = cell(N_objs,1);
%     objs(1) = cellstr('alcyl_2ft_1');
%     objs(2) = cellstr('alcyl_2ft_2');
%     objs(3) = cellstr('alcyl_3ft_1');
%     objs(4) = cellstr('alpipe');
%     objs(5) = cellstr('aluxo');
%     objs(6) = cellstr('bullet_105mm');
%     objs(7) = cellstr('ssuxo');
    objs(1) = cellstr('alcyl_2ft_1530');
    objs(2) = cellstr('alcyl_3ft');
    objs(3) = cellstr('alpipe');
    objs(4) = cellstr('aluxo');
    objs(5) = cellstr('bullet_105mm_air');
    objs(6) = cellstr('bullet_105mm_h2o');
    objs(7) = cellstr('howitzer_cap_air');
    objs(8) = cellstr('howitzer_cap_h2o');
    objs(9) = cellstr('howitzer_nocap');
    objs(10) = cellstr('ssuxo');

    objData = zeros(length(objs),3);
    % FFN file parameters used in modeling with halfspace.exe
    % [Target name, radius, length, and radii ratio]
    objData(1,:) = [0.1524 0.6 0.9];
    objData(2,:) = [0.1524 0.6 0.9];
    objData(3,:) = [0.1619 0.9 0.9];
    objData(4,:) = [0.05347 0.6 0.9];
    objData(5,:) = [0.0525 0.6 0.9];
    objData(6,:) = [0.0525 0.6 0.9];
    objData(7,:) = [0.0775 0.6 0.9];
    objData(8,:) = [0.0775 0.6 0.9];
    objData(9,:) = [0.0775 0.6 0.9];
    objData(10,:) = [0.05347 0.6 0.9];

    %% Dummy Files to copy to directories which will be used in generating ACP's
    ins = 'C:\Users\halljj2\Desktop\WMSC-CODE\Synth_AC_Tools\ins';
    exes = 'C:\Users\halljj2\Desktop\WMSC-CODE\Synth_AC_Tools\exes';
    
%     %% Populating ins with necessary LSAS run files to be copied by buildDirs
%     cd(ins);
%     [rows cols] = size(envs)
%     % Creating novel LSAS runs for temp_in
%     for e = 1:rows
%         if ~find(envs(1:e-1,3)==envs(e,3))
%         run_el = createLsasRun(400,envs(e,3),10);
%         save(['lsas.10m_' num2str(envs(e,3)) '.dat'], run_el);
%         end
%     end
%     cd('..\')
    
    %% Creating the Directories in 'dir' which we will populate with dummy .in files
    dirMap = buildDirsLsas(dir,exes,ins, envs ,objs(obs), ranges , rots, runlen);

    %% Fixing Dummy Data to have meaningful values
    %  Fixing the .in files with the changed data; a non-fruitful fxn
    fixInsLsas(dirMap,envs,objs(obs),objData(obs,:),ranges,rots,chirp, runlen);

    %% Generating ACP's
    % Generating the database
    makeACPData(dirMap);
    cd(home);
end

