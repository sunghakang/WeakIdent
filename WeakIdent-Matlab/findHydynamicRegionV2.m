function [idxHighlyDynamic] = findHydynamicRegionV2(dictList,indOfInterestingFeatures,SandbLarge, numOfBins)
% script for finding highly dynamic region
% Copyright 2022, All Rights Reserved
% Code by Mengyi Tang
% For Paper, "WeakIdent: Weak formulation for Identifying
% Differential Equation using Narrow-fit and Trimming"
% by Mengyi Tang, Wenjing Liao, Rachel Kuske and Sung Ha Kang


[~, idx]             = ismember(dictList,indOfInterestingFeatures,'rows');
rescaleColInd        = idx >= 1;
scalesSums           = (sum(abs(SandbLarge(:,rescaleColInd)),2));

[~, idx]             = sort(scalesSums);
[mathbbNxmathbbNt,~] = histcounts(scalesSums,numOfBins);
inds                 = [1,cumsum(mathbbNxmathbbNt)];
transitionGroupIdx   = twoPieceFit(cumsum(mathbbNxmathbbNt));
idxHighlyDynamic     = idx(inds(transitionGroupIdx):inds(end));

end


function [tstarind] = twoPieceFit(y)
% twopiece_fit find the index of one main changing point of a curve y
% This scripted is modified from the code for Paper, "Weak SINDy for Partial Differential Equations"
% by D. A. Messenger and D. M. Bortz
NN = length(y);
y  = reshape(y,[1,length(y)]);
xx = 1:NN;

for k=2:NN-1
    subinds1 = 1:k;
    subinds2 = k:NN;
    y_av1 = y(subinds1);
    y_av2 = y(subinds2);
    m1 = range(y_av1)/range(xx(subinds1));
    m2 = range(y_av2)/range(xx(subinds2));
    L1 = min(y_av1)+m1*(xx(subinds1)-xx(1));
    L2 = max(y_av2)+m2*(xx(subinds2)-xx(end));
    errs(k-1) = sqrt(sum(((L1-y_av1)./y_av1).^2) + sum(((L2-y_av2)./y_av2).^2)); % relative l2
end
[~,tstarind] = min(errs);

end


