function [mx,mt,px,pt] = findcornersV2(U_obs,xs ,max_dx)
% This scripted is modified from the code for Paper, "Weak SINDy for Partial Differential Equations"
% by D. A. Messenger and D. M. Bortz

mx = [];
mt = [];
px = 0; pt = 0; maxd = 1;
if size(U_obs{1},1) > 1
    T = length(xs{end}); maxd = length(xs)-1;
end
tau = 1e-10;
tauhat = 2;

L = length(xs{1});
corners_all = cell(length(U_obs),1);

l = @(m,k,N) log((2*m-1)./m.^2).*(4*pi^2*k^2*m.^2-3*N^2*tauhat^2)-2*N^2*tauhat^2*log(tau);

for n= 1:length(U_obs)
   
    [corners] = findcornerpts(U_obs{n},xs);
    
    for d = 1: maxd
        N = length(xs{d});
        k =  corners{d}(2) ;%%%
        
        mstar1 = sqrt(3)/pi*N/2/k*tauhat;
        mstar2 = 1/pi*tauhat*(N/2)/k*sqrt(log(exp(1)^3/tau^8));
        mnew = fzero(@(m)l(m,k,N), [mstar1 mstar2]);
        if mnew>N/2-1
            mnew = N/2/k;
        end
        
        mx = [mx mnew];
        
        L = min(L,N);
    end
    k = corners{end}(2);
    
    if size(U_obs{1},1) > 1
       
        mnew = fzero(@(m)l(m,k,T), [1 2/sqrt(tau)]); % provably a bracket for [1 (1+sqrt(1-tol))/tol] for all j < certain value
        if mnew>T/2-1
            mnew = T/2/k;
        end
        mt = [mt mnew];
    end
    corners_all{n}=corners;
    
end

mx = min(floor((L-1)/2),ceil(mean(mx)));
px = max(max_dx+2,floor(log(tau)/log(1-(1-1/mx)^2)));
if size(U_obs{1},1) > 1
    mt = min(floor((T-1)/2),ceil(mean(mt)));
    pt = max(1+2,floor(log(tau)/log(1-(1-1/mt)^2)));
end


end

function [corners] = findcornerpts(U_obs,xs)

if size(U_obs,1) == 1
    dim = 1;
else
    dims = size(U_obs);
    dim = length(dims);
end

corners = cell(dim,1);
for d=1:dim
    if dim ==1
        shift = [1 2];
    else
        shift = circshift(1:dim,1-d);
        dim_perm = dims(shift);
    end
    
    x = xs{d}(:);
    L = length(x);
    wn = ((0:L-1)-floor(L/2))'*(2*pi)/range(x);
    xx = wn(1:ceil(end/2));
    NN = length(xx);
    Ufft = abs(fftshift(fft(permute(U_obs,shift))));
    if dim>2
        Ufft = reshape(Ufft,[L,prod(dim_perm(2:end))]);
        
    end
    if dim > 1
        Ufft = mean(Ufft,2);
    end
    if dim == 1
        Ufft = Ufft';
    end
    Ufft = cumsum(Ufft);
    
    Ufft = Ufft(1:ceil(L/2),:);
    errs = zeros(NN-6,1);
    for k=4:NN-3
        subinds1 = 1:k;
        subinds2 = k:NN;
        Ufft_av1 = Ufft(subinds1);
        Ufft_av2 = Ufft(subinds2);
        m1 = range(Ufft_av1)/range(xx(subinds1));
        m2 = range(Ufft_av2)/range(xx(subinds2));
        L1 = min(Ufft_av1)+m1*(xx(subinds1)-xx(1));
        L2 = max(Ufft_av2)+m2*(xx(subinds2)-xx(end));
        errs(k-3) = sqrt(sum(((L1-Ufft_av1)./Ufft_av1).^2) + sum(((L2-Ufft_av2)./Ufft_av2).^2)); % relative l2
    end
    [~,tstarind] = min(errs);
    tstar = -xx(tstarind);
    corners{d} = [tstar (NN-tstarind-3) ];
end
end