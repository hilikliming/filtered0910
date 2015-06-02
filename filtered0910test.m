%% This is a script for testing and comparing the performance of the LP-KSVD
% method versus the K-SVD and standard SVD method for manifold
% approximation given AC data signal type

clear all;
%clc;
home    =   cd;
above   =   '../';
cd(above);
dirm    =   'TestMATS'; % Directory where various matrices are saved (above repository)
%mkdir(dirm);
cd(dirm);
dirm    =   cd;
cd(home);

%% Generating FRM database w/ generateDatabaseLsas.m saving dirMap
% % !!!NOTE: Lines 57-58 of generateDatabaseLsas.m and 48 of fixInsLsas.m are 
% % hard-coded, change them to your local directories!

% cd(above);
% dirFRM  = 'DBFRM';
% mkdir(dirFRM);
% cd(dirFRM);dirFRM = cd;
% % Parameters for generateDatabaseLsas indicates environment conditions,
% % ranges, and target rotations to be modeled
% ranges = 10;%9.5:0.5:10.5;
% %water and sediment sound speeds
% c_w = [1448,1456,1464,1530];
% c_s = [1694,1694,1694,1694];
% % rotations to model
% rots = [0:20:80,270:20:350];
% % environment parameters to model, water, sediment speed, interface elevation
% envs      = zeros(length(c_w),3);
% envs(:,1) = c_w;
% envs(:,2) = c_s;
% envs(:,3) = 3.8*ones(length(c_w),1);
% % which of the 7 .ffn's to model
% objs    = [4,10,3,1]; 
% f_s     = 100e3;
% chirp   = [1 31 f_s]; % start and end freq of chirp defines center and BW, last number is f_s
% runlen  = [20,800]; %length meters, stops
% 
% cd(home);
% dirMapDBFRM = generateDatabaseLsas(dirFRM,envs,ranges,rots,objs,chirp,runlen);
% cd(dirm);
% save('dirMapDBFRM.mat','dirMapDBFRM');

cd(dirm);
load('dirMapDBFRM.mat');
cd(home);

%% Forming OBSERVATION (Testing) Matrix of all usable parts of the Filtered Runs

trials = 1:2;
resSVD  = zeros(length(trials),2);
resKSVD = zeros(length(trials),2);
resLP   = zeros(length(trials),2);
STATSSVD  = struct([]);
STATSKSVD = struct([]);
STATSLP   = struct([]);

for test = trials
stops = 800;
realTarg = {'AL_UXO_SHELL','STEEL_UXO_SHELL',... % 1,2
    'AL_PIPE','SOLID_AL_CYLINDER','ROCK1','ROCK2'}; % 3,4,5,6
[Y, t_Y, Dclutter] = realACfetch0910(realTarg,stops,6.75); % !!!This script has the
% 'UW Pond' Directory hardcoded in, change it (line 13 of realACfetch0910)
% to your Target Data dir!!!

%Y = Y*(eye(K)-ones(K,1)*ones(K,1)'/K);
cd(dirm);
save('Y.mat','Y');
save('t_Y.mat','t_Y');
save('Dclutter.mat','Dclutter');

cd(dirm);
load('Y.mat');
load('t_Y.mat');
load('Dclutter.mat');
cd(home);

%% Opening and Partitioning Rock Data

% % Shuffling Clutter Aspects
% Dclutter = Dclutter(:,randperm(size(Dclutter,2)));
% 
% % Splitting Clutter samples
% DcTrain = Dclutter(:,1:size(Dclutter,2)/2);
% DcTest = Dclutter(:,size(Dclutter,2)/2+1:end);
% 
% cd(dirm);
% save('DcTrain.mat','DcTrain');
% save('DcTest.mat','DcTest');

cd(dirm);
load('DcTrain.mat');
load('DcTest.mat');

cd(home);
Y = [Y DcTest];
t_Y = [t_Y;(max(t_Y)+1)*ones(size(DcTest,2),1)];


%% Extracting and Sampling Training Templates

% Ytrain       = getTrainingSamples(dirMapDBFRM(3:4,:,:,:),800);
% Ytrain(5).D  = DcTrain;
% 
% cd(dirm);
% save('Ytrain.mat','Ytrain'); 

cd(dirm);
load('Ytrain.mat');
cd(home);
 
%% Sub-sampling and organizing our training data to two classes for training

% YtrainSub = struct([]);
% 
  pickDs  = [1,2;3,4];
% 
% for m = 1:size(pickDs,1)
%     DD =[];
%     for c = pickDs(m,:)
%         if(c>0)
%             if(c~=5)
%                 pick = randsample(size(Ytrain(c).D,2),floor(1/7*size(Ytrain(c).D,2))); % 2 envs/obj
%             else
%                 pick = randsample(size(Ytrain(c).D,2),floor(1/10*size(Ytrain(c).D,2)));
%             end
%         D = Ytrain(c).D;
%         D = D(:,pick);
%         DD = [DD,D];
%         end
%     end
%     YtrainSub(m).D= DD;
% 
% end
% 
% cd(dirm);
% save('YtrainSub.mat','YtrainSub');

cd(dirm);
load('YtrainSub.mat');
cd(home);

%% RESIZE SIGNAL to only match on good frequencies (1-31kHz)

low_f  = 11; % ...no chirp during [0-1kHz)
high_f = 305; % Using up to 30 kHz (beyond this is mostly 0 so nuissance params for LS)
Y      = Y(low_f:high_f,:);

% Doing same to our training data
for m = 1:size(pickDs,1)
    D = YtrainSub(m).D;
    D = D(low_f:high_f,:);
    YtrainSub(m).D = D;
end

% % %
N = size(Y,1); % Signal length is N, # test samples is K
cd(home);

%% Training signal subspaces via SVD/K-SVD/LP-KSVD with same run parameters

% param.numIteration          = 10; % number of iterations to perform (paper uses 80 for 1500 20-D vectors)
% param.preserveDCAtom        = 0;
% param.InitializationMethod  = 'DataElements';
% param.displayProgress       = 1;
% param.minFracObs            = -.1; % min % of observations an atom must contribute to, else it is replaced
% param.maxIP                 = 0.995; % maximum inner product allowed betweem atoms, else it is replaced
% param.coeffCutoff           = 0.001; % cutoff for coefficient magnitude to consider it as contributing
% 
% % Parameters related to sparse coding stage
% coding.denoise_gamma = 0.1;
% coding.method    = 'MP';
% coding.errorFlag = 1;            
% coding.errorGoal = 1e-4; % allowed representation error for each signal (only if errorFlag = 1)
% coding.eta       = 1e-6; % Used in LP-KSVD
% coding.tau       = 10;   % Used in OMP during Sparse coding phase of KSVD
% coding.L         = coding.tau;
% 
% D_SVD  = struct([]);
% D_KSVD = struct([]);
% D_KL   = struct([]);
% 
% R_m     = struct([]);
% mu_m    = struct([]);
%     
% %D_LP   = struct([]);
% 
% 
% DD   =[];% Used in collecting random samples from each class to start LP-KSVD dictionary
% Data =[];% Used in collecting training samples from each class for LP-KSVD
% 
% ms   =[700,700,700,700]%[150,150,150,150,15]; % number of atoms to train for each class
% % Creating SVD and KSVD dictionaries and accumulating KSVD atoms for 
% % LP-KSVD joint solution
% 
% sdsz=0 % size of seed (real data used to help training samples) set to 0 for FRM only
% % pcs = 1:8;
% for m = 1:size(pickDs,1) % for each row in pickDs (which groups objects into m classes)
%     D = YtrainSub(m).D; 
%     % Adding Real data content if sdsz>0
%     if sdsz >0
%         P = [];
%         for c = pickDs(m,:) % using the classes in that row to seed the training samples
%             P = [ P Y(:,t_Y==m)];
%         end
% 
%         els = randsample(size(P,2),sdsz);
%         D   = [D,P(:,els)];
%     end
%     
%     %K-SVD Training
%     param.K  = ms(m); % ms(m) tells how many atoms to train in class m
%     [Dk,out] = KSVD(D,param,coding);
%     Dk = orderDict(Dk,out.CoefMatrix);   
%     D_KSVD(m).D = Dk;
%     
%     
%     % SVD Training
%     [U,S,V]     = svd(D,'econ');
%     D_SVD(m).D  = U(:,1:N);%R_m(m).R*(U(:,1:ms(m))-mu_m(m).mu*ones(1,ms(m)));
%     
% %     % KL Transform
% %     mu_m(m).mu = mean(D,2);
% %     R         = cov((D)');
% %     [U,~,~]   = svd(R);
% %     R_m(m).R  = U(:,pcs)';
% %     Uu        = U(pcs,pcs);%*(D-mu_m(m).mu*ones(1,size(D,2)));
% %     D_KL(m).D = Uu;
%     
%     % LP-KSVD Training Setup
%     Data = [Data,D]; % Collecting all training samples in group for LP-KSVD joint solution
%     DD   = [DD,D(:,randsample(size(D,2),ms(m)));]; %seeding LP-KSVD dict with random vectors from class
%     
% end
% % % De = Ytrain(5).D;
% % % [U S V]=svd(De(low_f:high_f,:));
% % % D_SVD(2).D = [D_SVD(2).D U(:,1:10)];
% % % D_KSVD(2).D = [D_KSVD(2).D U(:,1:10)];
% 
% % LP-KSVD Training on block matrix of Training samples (i.e. 'Data')
% param.K= size(DD,2);
% param.Dict = DD;
% 
% DDD = LPKSVD(Data,param,coding);
% b=1;
% for m=1:size(pickDs,1)
% D_LP(m).D = DDD(:,b:b+ms(m)-1);
% b= ms(m)+b;
% end
% 
% 
% 
% %% Saving the Learned Signal Subspaces
% 
% cd(dirm);
% %save('R_m.mat','R_m');
% %save('mu_m.mat','mu_m');
% save('D_SVD.mat','D_SVD');
% save('D_KSVD.mat','D_KSVD');
% % save('D_KL.mat','D_KL');
% save('D_LP.mat','D_LP');

cd(dirm);
load('R_m.mat','R_m');
load('mu_m.mat','mu_m');
load('D_SVD.mat'); 
load('D_KSVD.mat'); 
load('D_KL.mat'); 
load('D_LP.mat'); 


cd(home);
% NOT Using Prewhitening

for m = 1:size(pickDs,1)
    R_m(m).R=eye(N);
    mu11 = mu_m(m).mu;
    mu11 = mu11(1:N);
    mu_m(m).mu= (0)*mu11;
end


%% Trimming Down Various Dictionaries (Fine Tuning)

mSVD    =   [20 24]
mKSVD   =   [size(D_KSVD(1).D,2),size(D_KSVD(2).D,2)]%,size(D_KSVD(3).D,2),size(D_KSVD(4).D,2)]%[340 340 340 340]%
mLP     =   [size(D_LP(1).D,2),size(D_LP(2).D,2)]%[800 800]

% To account for discrepancy in lower frequencies of model (not currently
% in use hence the '[]'
h_bias = [];%zeros(301,1); 
%h_bias(1:10)=1; 
%h_bias(11)=1.3; 
%h_bias=h_bias/norm(h_bias);
 
% Trimming down and adding bias vector
for m = 1:size(pickDs,1)
    USVD  = D_SVD(m).D;
    UKSVD = D_KSVD(m).D;
    ULP   = D_LP(m).D;
    D_SVD(m).D  = [USVD(:,1:mSVD(m)) h_bias];
    D_KSVD(m).D = [UKSVD(:,1:mKSVD(m)) h_bias];
    D_LP(m).D   = [ULP(:,1:mLP(m)) h_bias];%[ULP(:,1:mLP(m)) h_bias];
end

%% Shuffling within class Test Observations

% for m = 1:numel(pickDs)
%      CY = Y(:,t_Y==m);
%      Y(:,t_Y==m) = CY(:,randperm(size(CY,2)));  
% end

% Removing Rocks
Y = Y(:,t_Y~=5);
t_Y=t_Y(t_Y~=5);
[N, K]=size(Y);

%% Running the WMSC
est  = 'MSD'
sigA  = 3 % Number of Aspects used per decision
tauK  = 2
tauLP = 1

d_YSVD  = WMSC(Y,D_SVD,mu_m,R_m,est,sigA);
d_YKSVD = OMPWMSC(Y,D_KSVD,mu_m,R_m,est,sigA,tauK);%WMSC(Y,D_KSVD,mu_m,R_m,est,sigA);%
d_YLP   = LocalWMSC(Y,D_LP,mu_m,R_m,est,sigA,tauLP,1e-6);%WMSC(Y,D_LP,mu_m,R_m,est,sigA);

% %% Displaying Result/Method Comparison
% figure;
% plot(d_YSVD);
% title('SVD');
% legend('J_T','J_{NT}');%legend('J_1','J_2','J_3','J_4');%legend('J_1','J_2','J_3','J_4','J_C');%
% axis([0,size(Y,2),min(min(d_YSVD)),max(max(d_YSVD))]);
% 
% figure;
% plot(d_YKSVD);
% title('KSVD');
% legend('J_T','J_{NT}');%legend('J_1','J_2','J_3','J_4');%legend('J_1','J_2','J_3','J_4','J_C');%
% axis([0,size(Y,2),min(min(d_YKSVD)),max(max(d_YKSVD))]);
% 
% figure;
% plot(d_YLP);
% title('LP-KSVD');
% legend('J_1','J_2','J_3','J_4','J_C');%legend('J_T','J_{NT}');%
% axis([0,size(Y,2),min(min(d_YLP)),max(max(d_YLP))]);

%% Documenting Result/Performance for Analysis

%CONVERTING TRUTH TABLE TO BINARY DECISIONS
% UXO vs. non-UXO (last 4 are cylinder, pipe, rocks)
% First 2 are aluxo, ssuxo
origT = t_Y;
t_Y(t_Y==1|t_Y==2)= 1;
t_Y(t_Y~=1)= 0;

% Initializing min discriminant value vector for K observations and a
% decision vector m_P

jmin_TSVD  = zeros(K,1);
jmin_NTSVD = zeros(K,1);
% 
jmin_TKSVD  = zeros(K,1);
jmin_NTKSVD = zeros(K,1);

jmin_TLP  = zeros(K,1);
jmin_NTLP = zeros(K,1);


m_YSVD  = zeros(K,1);
m_YKSVD = zeros(K,1);
m_YLP   = zeros(K,1);


% Determine Minimal Discriminant Value to make decision and record decision
T  = [1];%[1];%[1,2];
NT = [2];%[2];%[3,4,5];

% Finding minimal value from UXO and non UXO families of classes
for k = 1:K
% Strict Class decision
[~, m_YSVD(k)] = min(d_YSVD(k,:));
% Used in ratio comparison test for binary classifier
jmin_TSVD(k) = min(d_YSVD(k,T));%/norm(d_YSVD(k,T));
jmin_NTSVD(k) = min(d_YSVD(k,NT));%/norm(d_YSVD(k,NT));

[~, m_YKSVD(k)] = min(d_YKSVD(k,:));
jmin_TKSVD(k) = min(d_YKSVD(k,T));%/norm(d_YKSVD(k,T));
jmin_NTKSVD(k) = min(d_YKSVD(k,NT));%/norm(d_YKSVD(k,NT));

[~, m_YLP(k)] = min(d_YLP(k,:));
jmin_TLP(k) = min(d_YLP(k,T));%/norm(d_Y(k,T));
jmin_NTLP(k) = min(d_YLP(k,NT));%/norm(d_Y(k,NT));
end

% Setting UXO class I for objects classified as 1,2 or 3, other non UXO
% detections become non UXO class 0


origMSVD = m_YSVD;
origMKSVD = m_YKSVD;
origMLP = m_YLP;

for t = T

m_YSVD(m_YSVD==t) = 1;
m_YKSVD(m_YKSVD==t) = 1;
m_YLP(m_YLP==t) = 1;
end


m_YSVD(m_YSVD~=1)   = 0;
m_YKSVD(m_YKSVD~=1) = 0;
m_YLP(m_YLP~=1)     = 0;

%% ROC Curve Forming
gammas  = 0.2:1e-2:2;
gammas  = [gammas, 121];

P_dSVD     = zeros(length(gammas),1);
P_faSVD    = P_dSVD;

P_dKSVD    = P_dSVD;
P_faKSVD   = P_dSVD;

P_dLP     = zeros(length(gammas),1);
P_faLP    = P_dLP;

gam_i   = 1;

for gamma = gammas

    dSVD  = zeros(K,1);
    dKSVD = zeros(K,1);
    dLP   = zeros(K,1);
    for k = 1:K
        %If the J_m that lies closest to m subspace is still too great we
        %say it's a target

        dSVD(k)  = (jmin_TSVD(k)/jmin_NTSVD(k) < gamma);
        dKSVD(k) = (jmin_TKSVD(k)/jmin_NTKSVD(k) < gamma);
        dLP(k)   = (jmin_TLP(k)/jmin_NTLP(k) < gamma);
    end
    
    dSVD  = logical(dSVD);
    dKSVD = logical(dKSVD);
    dLP   = logical(dLP);
    
    
    P_dSVD(gam_i)   = sum(dSVD & logical(t_Y))/sum(logical(t_Y));
    P_faSVD(gam_i)  = sum(dSVD & ~logical(t_Y))/sum(~logical(t_Y));
    
    P_dKSVD(gam_i)  = sum(dKSVD & logical(t_Y))/sum(logical(t_Y));
    P_faKSVD(gam_i) = sum(dKSVD & ~logical(t_Y))/sum(~logical(t_Y));
    
    P_dLP(gam_i)    = sum(dLP & logical(t_Y))/sum(logical(t_Y));
    P_faLP(gam_i)   = sum(dLP & ~logical(t_Y))/sum(~logical(t_Y));
    
    gam_i = gam_i +1;
end

[~,gamkSVD]     = min(abs(1-(P_dSVD+P_faSVD)));
[~,gamkKSVD]    = min(abs(1-(P_dKSVD+P_faKSVD)));
[~,gamkLP]      = min(abs(1-(P_dLP+P_faLP)));

resSVD(test,:)  = [P_dSVD(gamkSVD), P_faSVD(gamkSVD)];
resKSVD(test,:) = [P_dKSVD(gamkKSVD), P_faKSVD(gamkKSVD)];
resLP(test,:)   = [P_dLP(gamkLP), P_faLP(gamkLP)];

STATSSVD(test).res = [P_faSVD P_dSVD ];
STATSKSVD(test).res= [P_faKSVD P_dKSVD];
STATSLP(test).res  = [P_faLP P_dLP];

STATSSVD(test).stats  = d_YSVD;
STATSKSVD(test).stats = d_YKSVD;
STATSLP(test).stats   = d_YLP;
end


%% AVERAGING TRIALS
P_faSVD   = 0*P_faSVD;
P_faKSVD  = 0*P_faKSVD;
P_faLP    = 0*P_faLP;
P_dSVD    = 0*P_faSVD;
P_dKSVD   = 0*P_faKSVD;
P_dLP     = 0*P_faLP;
for test = trials
    tSVD  = STATSSVD(test).res/length(trials);
    tKSVD = STATSKSVD(test).res/length(trials);
    tLP   = STATSLP(test).res/length(trials);
    
    P_faSVD = P_faSVD +tSVD(:,1);
    P_faKSVD = P_faKSVD+tKSVD(:,1);
    P_faLP = P_faLP+tLP(:,1);

    P_dSVD = P_dSVD +tSVD(:,2);
    P_dKSVD = P_dKSVD +tKSVD(:,2);
    P_dLP = P_dLP+tLP(:,2);
end

% Finding the averaged knee point
[~,gamkSVD]     = min(abs(1-(P_dSVD+P_faSVD)));
[~,gamkKSVD]    = min(abs(1-(P_dKSVD+P_faKSVD)));
[~,gamkLP]      = min(abs(1-(P_dLP+P_faLP)));

% Plotting ROCs and labeling Knee-Points
hndl=figure;
hold on
plot(P_faSVD,P_dSVD,P_faKSVD,P_dKSVD,P_faLP,P_dLP );%plot(P_faSVD,P_dSVD,P_faKSVD,P_dKSVD);%
legend('SVD', 'K-SVD','LP-KSVD');%legend('SVD','K-SVD');%
tag = ['ROC for Various Tests', 'Asp/obs = ', num2str(sigA)];
title(tag); xlabel('P_{FA} (%)'); ylabel('P_{CC} (%)');
plot([P_faSVD(gamkSVD), P_faKSVD(gamkKSVD),P_faLP(gamkLP)],[P_dSVD(gamkSVD),P_dKSVD(gamkKSVD),P_dLP(gamkLP)],'o');%plot([P_faSVD(gamkSVD),P_faKSVD(gamkKSVD)],[P_dSVD(gamkSVD),P_dKSVD(gamkKSVD)],'o');%
axis([0, 1, 0, 1]);
hold off
cd(dirm);
print(hndl,['0910fil',num2str(sigA),'aMSDhd.png'],'-dpng');
cd(home);



%% CREATING CONFUSION MATRICES
dSVD = zeros(K,1);
dKSVD = zeros(K,1);
dLP = zeros(K,1);

alpha = 0.02;

% 'Neyman-Pearson' Gamma that admits no more than alpha P_fa can be used
% for binary confusion matrix...
[~, gamnpSVD]   = min(abs(alpha-P_faSVD));
[~, gamnpKSVD]  = min(abs(alpha-P_faKSVD));
[~, gamnpLP]    = min(abs(alpha-P_faLP));

 for k = 1:K

 dSVD(k)    = jmin_TSVD(k)/jmin_NTSVD(k) < gammas(gamkSVD);
 dKSVD(k)   = jmin_TKSVD(k)/jmin_NTKSVD(k) < gammas(gamkKSVD);
 dLP(k)     = jmin_TLP(k)/jmin_NTLP(k) < gammas(gamkLP); %gammas(gamk);
 end
% decisions we made with a kneepoint gamma-------------^
m_YSVD  = dSVD;
m_YKSVD = dKSVD;
m_YLP   = dLP;

[CSVD, order]   = confusionmat(t_Y,m_YSVD);
[CKSVD, order]  = confusionmat(t_Y,m_YKSVD);
[CLP, order]    = confusionmat(t_Y,m_YLP);

% Normalizing Confusion Matrix
for j = 1:length(order)

    CSVD(j,:)   = CSVD(j,:)./sum(CSVD(j,:)+~any(CSVD(j,:)));
    CKSVD(j,:)  = CKSVD(j,:)./sum(CKSVD(j,:)+~any(CKSVD(j,:)));
    CLP(j,:)    = CLP(j,:)./sum(CLP(j,:)+~any(CLP(j,:)));
end

disp('SVD Binary Confusion Matrix:');
disp(CSVD);
disp('K-SVD Binary Confusion Matrix:');
disp(CKSVD);
disp('LP Binary Confusion Matrix:');
disp(CLP);

[CmSVD, order]  = confusionmat(origT,origMSVD);
[CmKSVD, order] = confusionmat(origT,origMKSVD);
[CmLP, order]   = confusionmat(origT,origMLP);

% Normalizing Confusion Matrix
for j = 1:length(order)
    
    CmSVD(j,:)  = CmSVD(j,:)./sum(CmSVD(j,:)+~any(CmSVD(j,:)));
    CmKSVD(j,:) = CmKSVD(j,:)./sum(CmKSVD(j,:)+~any(CmKSVD(j,:)));
    CmLP(j,:)   = CmLP(j,:)./sum(CmLP(j,:)+~any(CmLP(j,:)));
end

disp('SVD M-Class Confusion Matrix:');
disp(CmSVD);

disp('K-SVD M-Class Confusion Matrix:');
disp(CmKSVD);

disp('LP M-Class Confusion Matrix:');
disp(CmLP);




mean(resSVD,1)
mean(resKSVD,1)
mean(resLP,1)
