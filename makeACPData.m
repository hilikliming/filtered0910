function [ output_args ] = makeACPData(dirMap )
% This function generate acoustic color plots in directorys listed within
% dirMap given that these directories contain.in files for halfspace.in and
% acolor.in

for e = 1:length(dirMap(:,1,1,1))
    for o = 1:length(dirMap(1,:,1,1))
        for r = 1:length(dirMap(1,1,:,1))
            for a = 1:length(dirMap(1,1,1,:))
                dir = char(dirMap(e,o,r,a))
                cd(dir);
                system('..\halfspace.exe');
                system('..\acolor.exe');
                system('..\pcb.exe');
                system('..\tdi.exe');
                sax2mat('halfspace.in', 'acolor.in', 'tdi.in');
            end
        end
    end
end