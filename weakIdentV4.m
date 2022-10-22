function [cPred, cTrue, tableOfEqn, tableOfErr, time] = weakIdentV4(Uhat, xs, SSInX, alphaBar, polys, Tau,  ...
    trueCoefficients, useCrossDerivative, SSInT, IC, useErrDyn)
% WEAKIDENTV4: script for predicting PDE equations using WeakIdent

% Copyright 2022, All Rights Reserved
% Code by Mengyi Tang
% For Paper, "WeakIdent: Weak formulation for Identifying
% Differential Equation using Narrow-fit and Trimming"
% by Mengyi Tang, Wenjing Liao, Rachel Kuske and Sung Ha Kang

% INPUT:
% Uhat                : noisy data
% xs                  : {[x_1, x_2,...,],[t_1, t_2,...]}
% SSInX               : floor(mathbb{N}_x/boldsymbol{N}_x)
% SSInT               : floor(mathbb{N}_t/boldsymbol{N}_t)
% alphaBar            : highest order of derivative feature
% polys               : [0,1,...,betaBar] where betaBar is the highest
%                        order of polynomial power
% Tau                 : Trimming score
% useCrossDerivative  : 0/1 No/Yes use cross dimenstional derivative features;
% trueCoefficients    : true coefficients
% IC                  : initial condition (used in ODE to plot dynamic of predicted equation and compute dynamic error)
% useErrDyn           : 0/1 whether use dynamic error. (only applied to ode systems. In some cases this may slow down the computation.)

% OUTPUT:
% cPred               : predicted coefficients
% cTrue               : true coefficients (in terms of c) L*1 size vector
% tableOfEqn          : table of identified equations
% tableOfErr          : table of identification errors
% time                : cpu cost (second)

tic
numOfU            = length(Uhat);
mathbbNxmathbbNt  = size(Uhat{1});
dimXandT          = length(mathbbNxmathbbNt);
dimX              = dimXandT -1;
is1dODE           = 0;
dt                = xs{end}(2)-xs{end}(1);
if size(Uhat{1},1) == 1
    is1dODE = 1;
end

if is1dODE % ODE
    numOfBins         = 100;
    tspan             = xs{1}';
    options0          = odeset('RelTol',1e-10,'AbsTol',1e-10*ones(1,numOfU));
    lhsIdx            = [eye(numOfU), zeros(numOfU, 1), ones(numOfU,1)];
    dimX = 1;
    if length(Uhat) <=2
        sparsity = 10;
    else
        sparsity = 15;
    end
else % PDE
    numOfBins = 200;
    lhsIdx            = [eye(numOfU), zeros(numOfU, dimX), ones(numOfU,1)];
    if numOfU == 2 && dimX == 2
        sparsity      = 25;
    else
        sparsity      = 10;
    end
    dx                = xs{1}(2)-xs{1}(1); % in the case of 2 spatial domain, we assume dx = dy
end

[tags,dictList,lhsInd, cTrue] = getTagsV2(numOfU,dimXandT,lhsIdx,alphaBar,polys,useCrossDerivative,trueCoefficients);
if ~is1dODE
    [mx, mt, px, pt] = findcornersV2(Uhat, xs ,alphaBar);
else
    [mx, ~, px, ~]   = findcornersV2(Uhat, xs ,alphaBar);
end

if is1dODE == 1
    [WandbLarge, SandbLarge] = computeFeatureAndScaleMatrixODE( mx,  px,  SSInX,  dictList, Uhat, dt);
else
    [WandbLarge, SandbLarge] = computeFeatureAndScaleMatrix(mathbbNxmathbbNt, mx, mt, px, pt, SSInX, SSInT, dictList, Uhat, dx, dt, alphaBar);
end
indOfInterestingFeatures = findIdxOfInterestingFeaturesV2(numOfU, dimX, is1dODE); % dimX doesn't matter if is1dODE = 1
idxHighlyDynamic         = findHydynamicRegionV2(dictList,indOfInterestingFeatures, SandbLarge, numOfBins);
ScalesOfWandb            = mean(abs(SandbLarge(idxHighlyDynamic,:)),1);
L                        = size(WandbLarge,2);
scalesOfFeatures         = ScalesOfWandb(~ismember(1:L,lhsInd));
scalesOfFeatures         = 1./reshape(scalesOfFeatures,1, length(scalesOfFeatures));
scaleOfb                 = ScalesOfWandb(lhsInd);
W                        = WandbLarge(:,~ismember(1:L,lhsInd));
b                        = WandbLarge(:,lhsInd);
WTilda                   = W .*scalesOfFeatures;
bTilda                   = b .*scaleOfb;
bNarrowTilda             = bTilda(idxHighlyDynamic,:);
WNarrowTilda             = WTilda(idxHighlyDynamic,:);
cPred                    = zeros(size(WNarrowTilda,2),numOfU);

for num_var = 1:numOfU
    bLarge_       = b(:,num_var);
    bLargeTilda_  = bTilda(:,num_var);
    bNarrowTilda_ = bNarrowTilda(:,num_var);
    scaleOfb_     = scaleOfb(num_var);
    supportPred   = weakIdentFeatureSelectionV2(W,bLarge_, ...
        bLargeTilda_, WTilda, sparsity, scalesOfFeatures, scaleOfb_, ...
        WNarrowTilda, bNarrowTilda_, Tau);
    relativeScales = scalesOfFeatures' ./ scaleOfb(num_var);
    CoeffSP       = WNarrowTilda(:,supportPred) \ bNarrowTilda(:,num_var) .* relativeScales(supportPred);
    cPred(supportPred,num_var) = CoeffSP;
end

time         = toc;

% compute identification errors
e2           = computeL2NormErr(cTrue,cPred);
eInf         = computeErrInfty(cTrue,cPred);
[eTpr, ePpv] = computeTPRandPPVV2(cTrue,cPred);
eRes         = computeResidualErr(WNarrowTilda, cPred, relativeScales, bNarrowTilda);

if useErrDyn == 1
    [t_true, x_true ]              = ode45(@(t,x)ode_forward(x,cTrue,dictList(:,1:numOfU)),tspan,IC,options0);
    [t_weakIdent, x_weakIdent   ]  = ode45(@(t,x)ode_forward(x,cPred,dictList(:,1:numOfU)),tspan,IC,options0);
    row_max                        = min([length(t_true), length(t_weakIdent)]);
    tem                            = x_true(1:row_max,:) - x_weakIdent(1:row_max,:);
    err_dyn                        = mean(rms(tem,2));
    TypeOfError                    = {'$E_2$';'$E_{\infty}$';'$E_{res}$';'$E_{dyn}$';'$TPR$';'$PPV$'};
    WeakIdentResult                = [ e2; eInf; eRes; err_dyn; eTpr;  ePpv];
else
    TypeOfError     = {'$E_2$';'$E_{\infty}$';'$E_{res}$';'$TPR$';'$PPV$'};
    WeakIdentResult = [ e2; eInf; eRes; eTpr;    ePpv];
end
tableOfErr      = table(TypeOfError, WeakIdentResult);
tableOfEqn      = printEquations(cPred, tags,  lhsInd, cTrue, is1dODE);
end

function [errRes] = computeResidualErr(Wnarrow, c, errNorm, bNarrow)
errRes = norm(Wnarrow * (c./errNorm) - bNarrow) / norm(bNarrow);
end

function [tpr, ppv] = computeTPRandPPVV2(cTrue, cPred)
cPred = cPred(:);
cTrue = cTrue(:);
p  = sum(cPred~=0);
tp = sum((cPred.*cTrue) ~=0);
fp = p - tp;
t  = sum(cTrue ~= 0);
fn = t - tp;
tpr = tp/(tp + fn);
ppv = tp/(tp + fp);
end

function [errInf] = computeErrInfty(cTrue,c)
errInf = max(abs(c((cTrue~=0).*c ~=0) ./cTrue((cTrue~=0).*c ~=0) -1));
if isempty(errInf)
    errInf = NaN;
end
end
function [err2] = computeL2NormErr(cTrue,c)
err2 = norm(cTrue(:) - c(:)) / norm(cTrue(:));
end

function dy = ode_forward(y, ahat, tags)
% y:    initial condition,
% ahat: coefficients
[rowIdcs, ~] = find(ahat(:,:) ~=0);
rowIdcs = unique(rowIdcs);
ahat    = ahat(rowIdcs,:);
tags    = tags(rowIdcs,:);
yPool   = poolData_using_tages(y',length(y), tags);
dy      = (yPool * ahat)';
end

function yout = poolData_using_tages(yin, n_vars, tags)
% Copyright 2015, All Rights Reserved
% Code by Steven L. Brunton
% For Paper, "Discovering Governing Equations from Data:
%        Sparse Identification of Nonlinear Dynamical Systems"
% by S. L. Brunton, J. L. Proctor, and J. N. Kutz
yout = ones(1, size(tags,1));
for i = 1: size(tags,1)
    for j = 1:n_vars
        yout(1,i) = yout(1,i).* yin(1,j).^(tags(i,j));
    end
end
end

function yout = poolData(yin,nVars,polyorder,usesine)
% Copyright 2015, All Rights Reserved
% Code by Steven L. Brunton
% For Paper, "Discovering Governing Equations from Data:
%        Sparse Identification of Nonlinear Dynamical Systems"
% by S. L. Brunton, J. L. Proctor, and J. N. Kutz

n = size(yin,1);
% yout = zeros(n,1+nVars+(nVars*(nVars+1)/2)+(nVars*(nVars+1)*(nVars+2)/(2*3))+11);

ind = 1;
% poly order 0
yout(:,ind) = ones(n,1);
ind = ind+1;

% poly order 1
for i=1:nVars
    yout(:,ind) = yin(:,i);
    ind = ind+1;
end

if(polyorder>=2)
    % poly order 2
    for i=1:nVars
        for j=i:nVars
            yout(:,ind) = yin(:,i).*yin(:,j);
            ind = ind+1;
        end
    end
end

if(polyorder>=3)
    % poly order 3
    for i=1:nVars
        for j=i:nVars
            for k=j:nVars
                yout(:,ind) = yin(:,i).*yin(:,j).*yin(:,k);
                ind = ind+1;
            end
        end
    end
end

if(polyorder>=4)
    % poly order 4
    for i=1:nVars
        for j=i:nVars
            for k=j:nVars
                for l=k:nVars
                    yout(:,ind) = yin(:,i).*yin(:,j).*yin(:,k).*yin(:,l);
                    ind = ind+1;
                end
            end
        end
    end
end

if(polyorder>=5)
    % poly order 5
    for i=1:nVars
        for j=i:nVars
            for k=j:nVars
                for l=k:nVars
                    for m=l:nVars
                        yout(:,ind) = yin(:,i).*yin(:,j).*yin(:,k).*yin(:,l).*yin(:,m);
                        ind = ind+1;
                    end
                end
            end
        end
    end
end

if(usesine)
    for k=1:10
        yout = [yout sin(k*yin) cos(k*yin)];
    end
end
end