function [SLarge] = computeSLarge(W, dictList,Uhat, fftPhis,  subSamplingIdx, mx,mt,dx,dt)
% COMPUTESLARGE: script for computing the scale matrix
% Copyright 2022, All Rights Reserved
% Code by Mengyi Tang
% For Paper, "WeakIdent: Weak formulation for Identifying
% Differential Equation using Narrow-fit and Trimming"
% by Mengyi Tang, Wenjing Liao, Rachel Kuske and Sung Ha Kang

SLarge   = [];
ind      = 2;
dimXandT = length(size(Uhat{1}));
numOfU   = length(Uhat);

while ind < size(dictList,1)+1
    tags               = dictList(ind,1:numOfU);
    beta_u_plus_beta_v = sum(tags);
    if beta_u_plus_beta_v > 1
        if numOfU == 1
            fcn = (tags(1)) * ((Uhat{1} ).^(tags(1)-1));
        elseif numOfU == 2
            if tags(1) == 0
                fcn = (tags(2)) * ((Uhat{2} ).^(tags(2)-1));
            elseif tags(2) == 0
                fcn = (tags(1)) * ((Uhat{1} ).^(tags(1)-1));
            else
                fcn =       tags(1) * (Uhat{1} ).^(tags(1)-1) .* (Uhat{2}).^(tags(2));
                fcn = fcn + tags(2) * (Uhat{2} ).^(tags(2)-1) .* (Uhat{1}).^(tags(1));
            end
        end
    end
    while all(dictList(ind,1:numOfU) == tags)
        if beta_u_plus_beta_v > 1
            test_conv_cell = cell(1, dimXandT);
            scale_factor = 1;
            if  sum(dictList(ind,numOfU+1:numOfU+dimXandT)) == 0
                for k = 1:dimXandT
                    test_conv_cell{k}   = fftPhis{k}(1,:);
                end
            else
                for k = 1:dimXandT
                    test_conv_cell{k} = fftPhis{k}(dictList(ind, numOfU+k)+1 ,:);
                end
            end
            fcn_conv = convNDfft(fcn,test_conv_cell,subSamplingIdx,2);
            SLarge(:,ind) = fcn_conv(:);
        else
            SLarge(:,ind) = W(:,ind);
            
        end
        ind = ind+1;
        if ind > size(dictList,1)
            break
        end
    end
end
SLarge(:,1) = 1;

end

function [X] = convNDfft(X,cols,sub_inds,ver,sub_inds_full)
% convNDfft: compute ND convolution utilizing separability over 'valid' points,
% i.e. those that do not require zero-padding. Uses FFT to
% compute each 1D convolution.
% Copyright 2020, All Rights Reserved
% Code by Daniel A. Messenger
% For Paper, "Weak SINDy for Partial Differential Equations"
% by D. A. Messenger and D. M. Bortz
Ns = size(X);
dim = length(Ns);
for k=1:dim
    if ver==1
        col = cols{k}(:);
        n = length(col);
        col_ifft = fft([zeros(Ns(k)-n,1);col]);
    else
        col_ifft = cols{k}(:);
    end
    if dim ==1
        shift = [1 2];
        shift_back = shift;
    else
        shift = circshift(1:dim,1-k);
        shift_back=circshift(1:dim,k-1);
    end
    X = ifft(col_ifft.*fft(permute(X,shift)));
    inds = sub_inds(k);
    for j=2:dim
        inds{j} = ':';
    end
    X = X(inds{:});
    X = permute(X,shift_back);
end
X = real(X);
end

