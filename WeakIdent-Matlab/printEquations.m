function [tableOfEquations] = printEquations(cPred, tags,  lhsInd, cTrue, isODE)
% printEquation: script for printing equations from true and predicted
% coefficients

% Copyright 2022, All Rights Reserved
% Code by Mengyi Tang
% For Paper, "WeakIdent: Weak formulation for Identifying
% Differential Equation using Narrow-fit and Trimming"
% by Mengyi Tang, Wenjing Liao, Rachel Kuske and Sung Ha Kang

[L,numOfU]           = size(cPred);
eqnsWeakIdent        = cell(numOfU,1);
eqnsTrue             = cell(numOfU,1);

for k=1:numOfU
    tagsPdeFeatures  = tags(~ismember(1:L+numOfU,lhsInd));
    eqnsWeakIdent{k} = printEquationV3(cPred, k, tagsPdeFeatures,tags, lhsInd, isODE);
    eqnsTrue{k}      = printEquationV3(cTrue, k ,tagsPdeFeatures,tags, lhsInd, isODE);
end

if numOfU == 1
    TypeOfMethods = {'True';    'WeakIdent'; };
    Equations     = {strjoin(eqnsTrue{1});...
        strjoin(eqnsWeakIdent{1})};
elseif numOfU == 2
    TypeOfMethods = {'True'; '';   'WeakIdent'; ''};
    Equations     = {strjoin(eqnsTrue{1});...
        strjoin(eqnsTrue{2});...
        strjoin(eqnsWeakIdent{1});...
        strjoin(eqnsWeakIdent{2})};
elseif numOfU == 3
    TypeOfMethods = {'True'; ''; '';   'WeakIdent'; '';''};
    Equations     = {strjoin(eqnsTrue{1});...
        strjoin(eqnsTrue{2});...
        strjoin(eqnsTrue{3});...
        strjoin(eqnsWeakIdent{1});...
        strjoin(eqnsWeakIdent{2});...
        strjoin(eqnsWeakIdent{3})};
end

tableOfEquations = table(TypeOfMethods, Equations);
end

