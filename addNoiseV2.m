function [Uhat] = addNoiseV2(U,sigmaSNR,rngSeed)
% ADDNOISEV2: script for simulating noise on data
% Copyright 2022, All Rights Reserved
% Code by Mengyi Tang
% For Paper, "WeakIdent: Weak formulation for Identifying
% Differential Equation using Narrow-fit and Trimming"
% by Mengyi Tang, Wenjing Liao, Rachel Kuske and Sung Ha Kang
n = length(U); % number of variables
rng(rngSeed);
if sigmaSNR>0
    Uhat = cell(n,1);
    for k=1:n
        std      = rms(U{k}(:)- (max(U{k}(:)) + min(U{k}(:)))/2);
        Uhat{k}  = U{k} + normrnd(0, sigmaSNR*std, size(U{k}));
    end
else
    Uhat = U;
end
end
