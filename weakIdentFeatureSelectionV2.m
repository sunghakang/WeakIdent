function [supportPred]=weakIdentFeatureSelectionV2(W, b, bLargeTilda, WLargeTilda, ...
    s,  scalesOfFeatures, scaleOfb, WNarrowTilda, bNarrowTilda,  Tau)
% FEATURECROSSSELECTFUNV4: script for finding a support

% Copyright 2022, All Rights Reserved
% Code by Mengyi Tang
% For Paper, "WeakIdent: Weak formulation for Identifying
% Differential Equation using Narrow-fit and Trimming"
% by Mengyi Tang, Wenjing Liao, Rachel Kuske and Sung Ha Kang

% INPUT:
% W           : W matrix
% b           : b vector
% WTilda      : W tilda, rescale W using s(h,l) on for h in highly dynamic region
% bTilda      : b tilda, rescale b using s where s is avg|b(h)| for h in highly dynamic region
% WNarrowTilda: W tilda in highly dynamic region
% bNarrowTilda: b tilda in highly dynamic region
% Tau         : trimming score

% OUTPUT:
% supportPred: predicted support (index of features)


display_tex            = 0; % 0/1 dispay the trimming process
supportList            = cell(s,1);
crossValidationErrList = zeros(s,1);
WColumnNorm            = vecnorm(W);

for i=1:s
    support       = SPV2(W * diag(1./WColumnNorm), b ./norm(b,2),i);
    if display_tex  == 1
        disp(' ')
        disp(['SP -- Sparsity level is    : ', num2str(i)])
        disp(['SP -- Support (Coef index) : ', num2str(support')])
    end
    cPred         = narrowFit(WNarrowTilda, bNarrowTilda, support, scalesOfFeatures, scaleOfb);
    trimScore     = computeTrimScore(WColumnNorm, support, cPred);
    if display_tex  == 1
        disp(['The trimming scores are    : ', num2str(trimScore)])
    end
    supportList{i}   = support;
    %% compute the residual error
    crossValidationErrList(i) = computeCrossValidationErrV2(support, WLargeTilda, bLargeTilda);
    if display_tex  == 1
        disp(['The CV residual error is   : ', num2str(crossValidationErrList(i))])
    end
    %% shrink the index if necessary
    while length(support) > 1
        idxOfLeastImportantFeature = (trimScore == min(trimScore));
        if min(trimScore) > Tau
            break
        end
        while sum(idxOfLeastImportantFeature) > 1
            temp = find(idxOfLeastImportantFeature == 1);
            idxOfLeastImportantFeature(temp(1)) = 0;
        end
        support(idxOfLeastImportantFeature)  = []; % delete this importance
        cPred         = narrowFit(WNarrowTilda, bNarrowTilda, support, scalesOfFeatures, scaleOfb);
        trimScore     = computeTrimScore(WColumnNorm, support, cPred);
        if display_tex  == 1
            disp(' ')
            disp(['For a smaller support (Coef index) : ', num2str(support')])
            disp(['Trimming score                     : ', num2str(trimScore)]);
        end
        crossValidationErrList(i) = computeCrossValidationErrV2(support, WLargeTilda, bLargeTilda);
        supportList{i}     = support;
        if display_tex  == 1
            disp(['We update the current smallest residual error ',...
                'for sparsity ',num2str(i), ' to be ' , num2str(crossValidationErrList(i))] );
        end
    end
end
[~, CrossIdx]=min(crossValidationErrList);
supportPred = supportList{CrossIdx}';

end


function [cPred] = narrowFit(W, b, A, scalesOfFeatures, scaleOfb)
% NARROWFIT: compute the predicted coefficients from error normalized feature matrix
cPred = W(:,A)\b;
cPred = abs(cPred'.*scalesOfFeatures(A)./scaleOfb);
end


function [trimScore] = computeTrimScore(WcolumnNorm, support, cPred)
% computeTrimScore: compute the trimming score from WColumnNorm and predicted coefficients
trimScore     = WcolumnNorm(support) .* cPred;
trimScore     = trimScore./max(trimScore);
end
