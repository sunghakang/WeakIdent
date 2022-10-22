clc
clear
close all

pde_num           = 2;
sigmaSNR          = 0.1; % relative signal-noise-ratio

pde_names     = {'Linear2d','VanderPol','Duffing', 'LotkaVolterra', 'Lorenz', 'heat', 'transportDiff', 'KdV.mat','KS', 'NLS', 'PM', 'rxnDiff'};
disp('---------------------------------------------------------------------------------------------')
pde_name    = pde_names{pde_num};
load(['datasetV2/', pde_names{pde_num}])

numOfU      = length(U);

disp(['Prediction for ',pde_name])
%%
rng('shuffle');
rng_seed = rng().Seed;
rng_seed = 123;
rng(rng_seed);

Uhat = addNoiseV2(U,sigmaSNR,rng_seed);
if ~exist('xh','var')
    if size(Uhat{1},1) == 1 % ODE equations
        th = floor(length(xs{end})/1000);
        xh = floor(length(xs{1})/1000);
    else % for PDE equations
        th = floor(length(xs{end})/100);
        xh = floor(length(xs{1})/100);
    end
end
if ~exist('Tau','var')
    Tau = 0.05;
end
if ~exist('useCrossDerivative', 'var')
    useCrossDerivative = 0;
end
if ~exist('useErrDyn', 'var')
    useErrDyn = 0;
end
if ~exist('IC', 'var')
    IC = [];
end
if ~exist('max_dx', 'var')
    max_dx = 0;
end


%%
[c, cTrue, tableEqn, tableErr, time] = weakIdentV4(Uhat, xs, xh,  max_dx, polys,Tau,  trueCoefficients,useCrossDerivative,  th,IC, useErrDyn);
disp(['Running time: ', num2str(time), 'seconds'])

tableEqn
tableErr
