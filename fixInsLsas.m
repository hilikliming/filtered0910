function [ argOut ] = fixInsLsas(dirMap,envs,objs,objData,ranges,aspects,chirp,runlen )
% This functions takes various range and aspect values for these objects and
% replaces them in the halfspace.in, acolor.in, and tdi.in files at
% appropriate lines
c_w = envs(:,1);
c_s = envs(:,2);
elv = envs(:,3);
BW = chirp(2)-chirp(1);
[nenvs, ~] = size(envs);
home = cd;
for e = 1:nenvs
    for o = 1:length(objs)
        for r = 1:length(ranges)
            for a = 1:length(aspects)
                dir = char(dirMap(e,o,r,a));
                cd(dir);
                hs = cell(17,1);
                ac = cell(10,1);
                td = cell(22,1);
                pc = cell(8,1);
                
                % Creating new contents for acolor.in
                ac{1} = 'F'; %Verbose?
                ac{2} = 'F';
                ac{3} = ['ENV' num2str(e) '_' char(objs(o)) '_' num2str(ranges(r)) '_' num2str(aspects(a))];
                ac{4} = ['ENV' num2str(e) '_' char(objs(o)) '_' num2str(ranges(r)) '_' num2str(aspects(a)) '.sax'];
                ac{5} = ['ENV' num2str(e) '_' char(objs(o)) '_' num2str(ranges(r)) '_' num2str(aspects(a)) '.replica'];
                ac{6} = ['lsas.' num2str(runlen(1)) 'm.dat'];
                ac{7} = [num2str(ranges(r)) ' 0 ' num2str(objData(o,1)) ];
                ac{8} = [num2str(chirp(1)),' ',num2str(chirp(2)),' 200'];
                ac{9} = '721';  % Angle resolution in the templates provided by Dr. Williams
                ac{10}= 'T 0'; % Target rotational angle...
                
                % Creating new contents for halfspace.in
                hs{1} = 'F'; % Verbose
                hs{2} = 'F F'; % Dump, spectrum
                hs{3} = 'F'; % Output as single pings; otherwise edm packing
                hs{4} = ['ENV' num2str(e) '_' char(objs(o)) '_' num2str(ranges(r)) '_' num2str(aspects(a))]; % Filename prefix
                hs{5} = 'T T T T'; % Paths 1, 2, 3, 4 included
                hs{6} = [num2str(floor(chirp(3)/1000)),' ', num2str(floor(2000*(sqrt(elv(e)^2+(ranges(r))^2))/c_w(e))), ' 20' ]; % Samp. rate, time offset, window len [kHz,ms,ms]
                hs{7} = 'F'; % Transmitted signal in a data file
                hs{8} = [num2str(BW/2+chirp(1)),' ',num2str(BW),' 6 8 6 ']; % Carrier and BW collected from Pond Data BasicInfo
                hs{9} = ['1000 ' num2str(c_w(e)) ' 0.']; % Host: Density, sound speed, loss parameter
                hs{10}= 'F'; % T == edfm, F == fluid medium
                hs{11} = ['2000 ' num2str(c_s(e)) ' 0.008']; % Density, sound speed, loss parameter
                hs{12} = ['lsas.' num2str(runlen(1)) 'm.dat'];
                hs{13} = 'file';
                hs{14} = ['C:\Users\halljj2\Desktop\WMSC-CODE\Synth_AC_Tools\ffn\', char(objs(o)), '.ffn'];%['..\..\..\..\..\ffn\' char(objs(o)) '.ffn'];
                hs{15} = 'fe';
                hs{16} = [num2str(ranges(r)) ' 0. ' num2str(objData(o,1)) ];  % Target coordinate [m]
                hs{17} = [num2str(objData(o,1)) ' ' num2str(objData(o,2)) ' 0.9 ' num2str(aspects(a)) ];  % Target radius, length, and radii ratio
                
                % Creating new contents for tdi.in
                td{1} = 'F';                    % Verbose
                td{2} = '0 0.5';                % Autofocus iterations (don't use)
                td{3} = 'F' ;                   % Dump intermediate stuff
                td{4} = 'F';                    % Output beam patterns
                td{5} = 'F 0.1 1 1 0';          % Holographic image (don't use)
                td{6} = 'F';                    % Interferometry output
                td{7} = 'F';                    % 2D FFT of image
                td{8} = 'F 10 10';              % Mag. and phase of pixel (px,py)
                td{9} = ['ENV' num2str(e) '_' char(objs(o)) '_' num2str(ranges(r)) '_' num2str(aspects(a)) '.pcb.dat'];  % pcb.dat file produced from a *.sax file
                td{10} = ['lsas.' num2str(runlen(1)) 'm.dat'];
                td{11} = ['ENV' num2str(e) '_' char(objs(o)) '_' num2str(ranges(r)) '_' num2str(aspects(a))];         % Prefix of output filenames
                td{12} = [ num2str(ranges(r)) ' 0 0']; % Coordinates of image center
                td{13} = '2 2'  ;               % Length and width of image
                td{14} = '200 200';             % Number of points in image
                td{15} = '4';                   % Interpolation points
                td{16} = [num2str(BW/2+chirp(1)),' ',num2str(BW)] ;              % Carrier frequency and bandwidth
                td{17} = ['1000 ' num2str(c_w(e))];      % Density and sound speed in water
                td{18} = 'o 0.1 0.1 20 0';      % Aperture info for rcv (don't use)
                td{19} = 'o 0.1 0.1 20 0';      % Aperture info for src (don't use)
                td{20} = '8.e-3';               % Time offset [s] from halfspace input
                td{21} = '2';                   % Spreading compensation (don't use)
                td{22} = 'F maskfile 0.1';      % mask mask_file threshold
                
                % Creating contents of new pcb.in file
                pc{1}='F';               % Verbose
                pc{2}='F';               % Dump intermediate stuff
                pc{3}=['ENV' num2str(e) '_' char(objs(o)) '_' num2str(ranges(r)) '_' num2str(aspects(a)) '.sax'];     % SAX file
                pc{4}=['ENV' num2str(e) '_' char(objs(o)) '_' num2str(ranges(r)) '_' num2str(aspects(a)) '.replica']; % Replica pulse file
                pc{5}=['ENV' num2str(e) '_' char(objs(o)) '_' num2str(ranges(r)) '_' num2str(aspects(a))];         % Prefix of output filenames
                pc{6}='16';              % Carrier frequency
                pc{7}='F 12 28 boxcar';  % want_filter, lo, hi, type_of_filter
                pc{8}='F 200';           % pcb for ping #
            
                % Writing the new contents
                fileID = fopen('acolor.in','w');
                fprintf(fileID,'%s\n',ac{:});
                fclose(fileID);
                
                fileID = fopen('halfspace.in','w');
                fprintf(fileID,'%s\n',hs{:});
                fclose(fileID);
                
                fileID = fopen('tdi.in','w');
                fprintf(fileID,'%s\n',td{:});
                fclose(fileID);
                
                fileID = fopen('pcb.in','w');
                fprintf(fileID,'%s\n',pc{:});
                fclose(fileID);

                
            end
        end
    end
end
cd(home);

end

