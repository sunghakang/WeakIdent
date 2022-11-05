function [WandbLarge, SandbLarge] = computeFeatureAndScaleMatrix(mathbbNxmathbbNt, mx, mt, px, pt, SSInX, SSInT, dictList, Uhat, dx, dt, alphaBar)
% computeFeatureAndScaleMatrix script for computing feature matrix and
% scale matrix

% Copyright 2022, All Rights Reserved
% Code by Mengyi Tang
% For Paper, "WeakIdent: Weak formulation for Identifying
% Differential Equation using Narrow-fit and Trimming"
% by Mengyi Tang, Wenjing Liao, Rachel Kuske and Sung Ha Kang


dimX = length(mathbbNxmathbbNt)-1;
subSamplingIdx = cell(1,length(mathbbNxmathbbNt));
shrinkSize     = [ones(1, dimX).* mx .* 2, mt*2];
SS             = [ones(1, dimX).* SSInX, SSInT];


for j=1:length(mathbbNxmathbbNt)
    subSamplingIdx{j} = 1:SS(j):mathbbNxmathbbNt(j)-shrinkSize(j);
end


[phiXs, phiTs] = computeTestFuns(mx, mt,px,pt, alphaBar);
fftPhis = computeFFTOfTestFuns(mathbbNxmathbbNt, phiXs, phiTs, mx, mt, dx, dt);
WandbLarge = computeWLarge(dictList,Uhat, fftPhis,  subSamplingIdx);
SandbLarge = computeSLarge(WandbLarge, dictList,Uhat, fftPhis,  subSamplingIdx, mx,mt,dx,dt);
% SandbLarge(SandbLarge==0) = 1;
end


function [phiXs, phiTs] = computeTestFuns(mx, mt, px,pt, alphaBar)
phiXs = computeDescretePhi(mx,alphaBar,px);
phiTs = computeDescretePhi(mt,1,pt);

end


function [fftPhis] = computeFFTOfTestFuns(mathbbNxmathbbNt, phiXs, phiTs, mx, mt, dx, dt)
% computeFFTOfTestFuns: script for computing the fft of test functions
% This scripted is modified from the code for Paper, "Weak SINDy for Partial Differential Equations"
% by D. A. Messenger and D. M. Bortz
dimXandT = length(mathbbNxmathbbNt);
fftPhis  = cell(dimXandT,1);
[mm,nn]   = size(phiXs);

for k=1:dimXandT-1
    fftPhis{k} = [zeros(mm,mathbbNxmathbbNt(k)-nn) (mx*dx ).^(-(0:mm-1)').*phiXs/nn ];
    fftPhis{k} = fft(fftPhis{k},[],2);
end

[mm,nn]       = size(phiTs);

fftPhis{dimXandT} = [zeros(mm,mathbbNxmathbbNt(dimXandT)-nn) (mt*dt ).^(-(0:mm-1)').*phiTs/nn];
fftPhis{dimXandT} = fft(fftPhis{dimXandT},[],2);

end







