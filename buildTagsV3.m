function [dicList] = buildTagsV3(numOfU,dimX,alphaBar,polys,useCrossDerivative, lhs)
% BUILDTAGSV3: build library tags for possible features

% Copyright 2022, All Rights Reserved
% Code by Mengyi Tang
% For Paper, "WeakIdent: Weak formulation for Identifying
% Differential Equation using Narrow-fit and Trimming"
% by Mengyi Tang, Wenjing Liao, Rachel Kuske and Sung Ha Kang

betas = zeros(0, numOfU);
for p = 2:length(polys)
    if numOfU == 3
        for i = 0:polys(p)
            for j = 0:polys(p)-i
                for k = polys(p)-i-j
                    betas = [betas; i,j,k];
                end
            end
        end
    elseif numOfU == 2
        for i = 0:polys(p)
            for j = polys(p)-i
                betas = [betas;i,j];
            end
        end
    elseif numOfU == 1
        betas = [betas; polys(p)];
    end
end

alphas = zeros(0, numOfU);
if useCrossDerivative == 1
    for p = 1:length(polys)
        if dimX == 3
            for i = 0:polys(p)
                for j = 0:polys(p)-i
                    for k = polys(p)-i-j
                        alphas = [alphas; i,j,k];
                    end
                end
            end
        elseif dimX == 2
            for i = 0:polys(p)
                for j = polys(p)-i
                    alphas = [alphas;i,j];
                end
            end
        elseif dimX == 1
            alphas = [alphas; polys(p)];
        end
    end
else
    alphas = zeros(1,dimX);
    for i = 1: dimX
        alphas(end+1:end+alphaBar, i) = [1:alphaBar]';
        
    end
end
%
dicList = [];
for i = 1:size(betas, 1)
    for j = 1:size(alphas,1)
        dicList = [dicList; betas(i,:), alphas(j,:)];
    end
end
dicList = [zeros(1, size(dicList,2)); dicList];
dicList(:,end+1) = 0;
dicList = [dicList; lhs];
dicList = sortrows(dicList);

end



