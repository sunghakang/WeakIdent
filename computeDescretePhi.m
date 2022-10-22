function [Cfs] = computeDescretePhi(m,d,p)
% COMPUTEDESCRETEPHI: get test function values on scaled reference grid,
% computed from max derivative (d) and half-support size m, such
% that 1D test function phi is supported on 2m+1 points

% This scripted is modified from code
% for Paper, "Weak SINDy for Partial Differential Equations"
% by D. A. Messenger and D. M. Bortz

t = (0:m)/m;
t_L = zeros(d+1,m+1);                             % store (1+t)^q, (1-t)^q
t_R = zeros(d+1,m+1);
for j=1:m
    t_L(:,j)  = (1+t(j)).^(fliplr(p-d:p))';
    t_R(:,j)  = (1-t(j)).^(fliplr(p-d:p))';
end

ps = ones(d+1,1);           % derivative coefficients
for q=1:d
    ps(q+1) = (p-q+1)*ps(q);
end
t_L = ps.*t_L;
t_R = ((-1).^(0:d)'.*ps).*t_R;

Cfs = zeros(d+1,2*m+1);      % Values of derivatives at grid points
Cfs(1,:) = [fliplr(t_L(1,:).*t_R(1,:)) t_L(1,2:end).*t_R(1,2:end)];
P = fliplr(pascal(d+1));
for k=1:d
    binoms = diag(P,d-k);
    Cfs_temp = zeros(1,m+1);
    for j=1:k+1
        Cfs_temp = Cfs_temp + binoms(j)*t_L(k+2-j,:).*t_R(j,:);
    end
    Cfs(k+1,:) = [(-1)^k*fliplr(Cfs_temp) Cfs_temp(2:end)];
end


end



