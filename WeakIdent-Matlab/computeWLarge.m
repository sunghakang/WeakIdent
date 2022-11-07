function [WLarge] = computeWLarge(dictList,Uhat, fftPhis,  subSamplingIdx)
% computeWLarge: script for computing feature matrix

% This scripted is modified from code
% for Paper, "Weak SINDy for Partial Differential Equations"
% by D. A. Messenger and D. M. Bortz

n = length(Uhat);
dim = length(size(Uhat{1}));
WLarge = [];
ind = 1;



while ind<size(dictList,1)+1
    tags = dictList(ind,1:n);
    if isreal(tags)
        fcn = (Uhat{1}).^tags(1);
        
        for k=2:n
            fcn = fcn.*((Uhat{k} ).^tags(k));
        end
        
    else
        ind_freq = find(imag(tags(1:n)));
        freq = sum(imag(tags(1:n)));
        if freq<0
            fcn = sin(abs(freq)*Uhat{ind_freq});
        else
            fcn = cos(abs(freq)*Uhat{ind_freq});
        end
    end
    
    while all(dictList(ind,1:n) == tags)
        test_conv_cell = {};
        for k=1:dim-1
            test_conv_cell{k} = fftPhis{k}(dictList(ind,n+k)+1,:);
        end
        test_conv_cell{dim} = fftPhis{dim}(dictList(ind,n+dim)+1,:);
        
        fcn_conv = convNDfft(fcn,test_conv_cell,subSamplingIdx,2);
        WLarge(:,ind) = fcn_conv(:);
        ind = ind+1;
        if ind > size(dictList,1)
            break
        end
        
    end
    
    
end

end

function [X] = convNDfft(X,cols,sub_inds,ver)
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
