function [ Dn ] = orderDict(D,Coef)
[~,map] = sort(sum(abs(Coef)>0,2));
map(map>size(D,2))=[];
D = D(:,map);
Dn = fliplr(D);

end

