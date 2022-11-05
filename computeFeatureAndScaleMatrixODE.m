function [WandbLarge, SandbLarge] = computeFeatureAndScaleMatrixODE( mx,  px,  SSInX,  dictList, Uhat, dx)
% computeFeatureAndScaleMatrix script for computing feature matrix and
% scale matrix

% Copyright 2022, All Rights Reserved
% Code by Mengyi Tang
% For Paper, "WeakIdent: Weak formulation for Identifying
% Differential Equation using Narrow-fit and Trimming"
% by Mengyi Tang, Wenjing Liao, Rachel Kuske and Sung Ha Kang

% This scripted is modified from the code for Paper, "Weak SINDy for Partial 
% Differential Equations" by D. A. Messenger and D. M. Bortz

Nt = size(Uhat{1},2);

subSamplingIdx = cell(1,2);

subSamplingIdx{1} = 1:SSInX:Nt-2*mx;
subSamplingIdx{2} = 1:SSInX:Nt-2*mx;

Cfs_x = computeDescretePhi(mx,1, px);

Cfs_ffts = {};

dim              = 1;
[mm,nn]          = size(Cfs_x);
Cfs_ffts{dim}    = [zeros(2,Nt-nn) (mx*dx ).^(-(0:mm-1)').*Cfs_x/nn];
Cfs_ffts{dim}    = fft(Cfs_ffts{dim},[],2);
Cfs_ffts{2}      = Cfs_ffts{1};
Cfs_ffts{1}(2,:) = [];

WandbLarge = computeWLarge1dODE(dictList,Uhat, Cfs_ffts,  subSamplingIdx);
SandbLarge = WandbLarge;

// SandbLarge(WandbLarge==0) = 1;

end


function [Theta_pdx] = computeWLarge1dODE(dictList, Uhat, Cfs_ffts, subSamplingIdx)
Theta_pdx = zeros(length(subSamplingIdx{1}), size(dictList,1) );
ind = 1;

numVar = length(Uhat);
while ind<size(dictList,1)+1
    
    tags = dictList(ind,1:numVar);
    u = (Uhat{1}).^tags(1);
    for k=2:numVar
        u = u.*((Uhat{k} ).^tags(k));
    end
    
    while all(dictList(ind,1:numVar) == tags)
        
        test_conv_cell = {};
        dim = 2;
        for k=1:dim-1
            test_conv_cell{k} = Cfs_ffts{k}(dictList(ind,numVar+k)+1,:);
        end
        test_conv_cell{dim} = Cfs_ffts{dim}(dictList(ind,numVar+dim)+1,:);
        
        fcn_conv = computeConv(u,test_conv_cell,subSamplingIdx);
        
        
        %         fftOfPhi   = Cfs_ffts{k}(dictList(ind,numVar+2)+1,:);
        %         fcn_conv         = conmputeConv(u,fftOfPhi,subSamplingIdx);
        Theta_pdx(:,ind) = fcn_conv(:);
        ind = ind+1;
        if ind > size(dictList,1)
            break
        end
    end
end

end

function [X] = computeConv(X,cols,subSamplingIdx)
X        = X';
col_ifft = cols{2}(:);
X        = ifft(col_ifft.*fft(X));
X        = X(subSamplingIdx{1}(:));
X        = real(X);
end


