function [err] = computeCrossValidationErrV2(ind, A, b)
% script for computing cross validation error
errCrossAccu = zeros(1,30);
for jj = 1:30
    e=computeCVErrV4(ind,A, b);
    errCrossAccu(jj) =  e;
end
err = mean(errCrossAccu)  + std(errCrossAccu);
end


function [err]=computeCVErrV4(support,W,b,ratio)
if ~exist('ratio','var')
    ratio = 1/100;
end

n     = size(b,1);
inds  = randperm(n);
W     = W(inds,:);
b     = b(inds);

% split the data into two parts
endOfPart1 = floor(n*ratio);

IdxPart1  = 1:endOfPart1;
IdxPart2  = endOfPart1+1:n;


% compute e1
coeff            = zeros(size(W,2),1);
coeff(support)   = W(IdxPart1,support)\b(IdxPart1);
e1               = norm(W(IdxPart2,:)*coeff - b(IdxPart2) ,2)/norm(b(IdxPart2),2);

% compute e2
coeff            = zeros(size(W,2),1);
coeff(support)   = W(IdxPart2,support)\b(IdxPart2);
e2               = norm(W(IdxPart1,:)*coeff - b(IdxPart1) ,2)/norm(b(IdxPart1),2);

% compute weighted error
err              = e1 * (1-ratio) + e2 * ratio;

end
