function [tagsPde,dicList,utIdx,cTrue] = getTagsV2(numOfU,dimXandT,lhsIdx,alphaBar,polys,useCrossDerivative,trueCoefficients)
% GETTAGSV2: script for find the index of intersting features

% Copyright 2022, All Rights Reserved
% Code by Mengyi Tang
% For Paper, "WeakIdent: Weak formulation for Identifying
% Differential Equation using Narrow-fit and Trimming"
% by Mengyi Tang, Wenjing Liao, Rachel Kuske and Sung Ha Kang

% Modified from code by Daniel A. Messenger for Paper, "Weak SINDy for Partial Differential Equations"

% INPUT:
% numOfU             : number of variables in a system
% dimXandT           : dimension of spatial domain X + 1
% lhsIdx             : tagIndex of u_t
% alphaBar           : highest order of derivative features
% polys              : [0,1,..., betaBar] -- power of monials
% useCrossDerivative : 0/1
% trueCoefficients   : true coeffficient values c with tags

% OUTPUT:
% tagsPde            : a list of tags of Pde
% dicList            : a list of features in the dictionary
% utIdx              : a list of index l of u_t/v_t...
% cTrue              : true sparse coefficient vector


[dicList]         = buildTagsV3(numOfU,dimXandT-1,alphaBar,polys,useCrossDerivative, lhsIdx);
[tagsPde,dicList] = buildStrTagsV2(dicList,dimXandT,numOfU);
utIdx             = zeros(size(lhsIdx,1),1);

for k=1:size(lhsIdx,1)
    utIdx(k) = find(ismember(dicList,lhsIdx(k,:),'rows'));
end


if ~isempty(trueCoefficients{1})
    try
        cTrue = zeros(size(dicList,1)-numOfU,numOfU);
        for k=1:numOfU
            axi_temp   = tags2axi(trueCoefficients{k},dicList);
            cTrue(:,k) = axi_temp(~ismember(1:size(dicList,1),utIdx));
        end
    catch
        u_input = input('True terms missing from library, proceed anyway?');
        if u_input~=1
            return;
        else
            cTrue = [];
        end
    end
else
    cTrue = [];
end
end


function true_nz_weights = tags2axi(true_nz_weight_tags,lib_list)
% TAGS2AXI: generate true model weights
% Copyright 2020, All Rights Reserved
% Code by Daniel A. Messenger
% For Paper, "Weak SINDy for Partial Differential Equations"
% by D. A. Messenger and D. M. Bortz
m                    = size(lib_list,1);
true_nz_weights      = zeros(m,1);
[~,loc]              = ismember(true_nz_weight_tags(:,1:end-1),lib_list,'rows');
true_nz_weights(loc) = true_nz_weight_tags(:,end);
end
