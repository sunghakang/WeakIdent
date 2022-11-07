function [idx] = findIdxOfInterestingFeaturesV2(numOfU,dimX, is1dODE)
% FINDIDXOFINTERESTINGFEATURES: script for find the index of intersting features

% Copyright 2022, All Rights Reserved
% Code by Mengyi Tang
% For Paper, "WeakIdent: Weak formulation for Identifying
% Differential Equation using Narrow-fit and Trimming"
% by Mengyi Tang, Wenjing Liao, Rachel Kuske and Sung Ha Kang

% INPUT:
% numOfU    : number of variables in a system
% dimX      : dimension of spatial domain X
% is1dODE   : 0/1 ode system

% OUTPUT:
% inds      : a list of index of interesting features;
%             idx = [ind1; ind2;...];
%             ind = [beta_1, ..., b_dimX, alpha_1,...,alpha_dimX, 0]

if is1dODE == 1
    idx = [eye(numOfU), zeros(numOfU, 2)];
else
    if numOfU == 1 && dimX == 1
        idx = [2,1,0];
        
    elseif numOfU == 1 && dimX == 2
        idx = [2,1,0,0; ...
            2,0,1,0; ...
            3,1,1,0];
        
    elseif numOfU == 2 && dimX == 1
        idx = [0,2,1,0; ...
            2,0,1,0];
        
    elseif numOfU == 2 && dimX == 2
        idx = [2, 0, 1, 0, 0;...
            2, 0, 0, 1, 0;...
            0, 2, 0, 1, 0;...
            0, 2, 1, 0, 0;...
            2, 1, 1, 0, 0;...
            1, 2, 0, 1, 0;...
            % 3, 0, 1, 1, 0;...  these features can be used for the case when cross-derivative features are allowed
            % 0, 3, 1, 1, 0;...
            ];
    end
end
end

