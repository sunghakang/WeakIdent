function support = SPV2(W,b,sparsity)
% "Subspace Pursuit for Compressive Sensing: Closing the
%  Gap Between Performance and Complexity"%

% INPUT
% W        :  feature matrix
% sparsity :  sparsity level

% OUTPUT
% support  :  a list of index

itermax = 15;
[~,N]=size(W);
ISDISP = 0;
%%
cv = abs( b'*W );
[~, cv_index] = sort(cv,'descend');

Lda = cv_index(1:sparsity);
Phi_Lda = W(:,Lda);
if(ISDISP)
    display(sort(Lda));
end



x = (Phi_Lda'*Phi_Lda)\(Phi_Lda' * b);
r = b - Phi_Lda*x;
res = norm(r);
iter = 0;
if(ISDISP); display(sprintf('iter=%d, res=%g', iter, res)); end
if (res < 1e-12)
    X = zeros(N,1);
    X(Lda)=x;
    support = find(X);
    return
end
usedlda = zeros(1,N);
usedlda(Lda)=1;
for iter = 1:itermax
    res_old = res;
    %%Step1 find T^{\prime} and add it to \hat{T}
    cv = abs( r'*W );
    [~, cv_index] = sort(cv,'descend');
    Sga = union(Lda, cv_index(1:sparsity));
    Phi_Sga = W(:,Sga);
    
    %% find the most significant K indices
    x_temp = (Phi_Sga'*Phi_Sga)\(Phi_Sga' * b);
    
    [~, x_temp_index] = sort( abs(x_temp) , 'descend' );
    Lda = Sga(x_temp_index(1:sparsity));
    Phi_Lda = W(:,Lda);
    if(ISDISP); display(sort(Lda)); end
    usedlda(Lda)=1;
    %% calculate the residue
    x = (Phi_Lda'*Phi_Lda)\(Phi_Lda' * b);
    r = b - Phi_Lda*x;
    res = norm(r);
    if(ISDISP); display(sprintf('iter=%d, res=%g', iter, res)); end
    X = zeros(N,1);
    X(Lda)=x;
    if ( res/res_old >= 1 || res < 1e-12)
        support = find(X);
        return
        sprintf('HIIIIIIIIIIIIIIIIIIIIIIIIIIIII')
    end
end
support = find(X);

