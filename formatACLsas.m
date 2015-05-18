function [ TSo ] = formatACLsas( AC, N_s )
%% Input:
%  AC - columns are aspects
%  N_s - Desired frequency bin number per aspect observation
%% Output:
%  TS - Target Strength AC with N_s frequency bins
%%%%%% AC = 20*log10(AC); no.
[N,K] = size(AC);
%TS = zeros(N_s,size(AC,2));
eps = 3;
TS = AC(:,sum(AC,1)>eps);
sub1 = AC(:,1:floor(K/2)); sub1 =sub1(:,sum(sub1,1)>eps);
sub2 = AC(:,floor(K/2):K); sub2 =sub2(:,sum(sub2,1)>eps);
bc = max([size(sub1,2),size(sub2,2)]);
TS = circshift(TS,bc,2);
TSo = zeros(N_s,size(TS,2));
for k = 1:size(TS,2)
TSo(:,k) = resample(TS(:,k),N_s,size(TS,1));
%TSo(:,k) = 20*log10(abs(TSo(:,k))/norm(TSo(:,k)));
end

end

