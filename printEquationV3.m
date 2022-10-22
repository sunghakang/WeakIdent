function [output] = printEquationV3(W,k, tagsPdeRdx, tagsPde, lhsIdx, isODE)
% printEquationForPdeV1: to print the PDE/ODE equations provided tags of features

% Copyright 2022, All Rights Reserved
% Code by Mengyi Tang
% For Paper, "WeakIdent: Weak formulation for Identifying
% Differential Equation using Narrow-fit and Trimming"
% by Mengyi Tang, Wenjing Liao, Rachel Kuske and Sung Ha Kang


x = printEqn(W(:,k),tagsPdeRdx,tagsPde{lhsIdx(k)});
if isempty(x)
    output = " ";
else
    output = x;
end
if isODE == 1
    output = renameEqnOde(output);
else
    output = renameEqnPde(output);
end


output(1)   = "$" + output(1);
output(end) = output(end) + "$";

end

function str = printEqn(w,tags_pde,lhs)
% code modified from print_pde for Paper, "Weak SINDy for Partial Differential Equations"
%%%%%%%%%%%% by D. A. Messenger and D. M. Bortz
nz_inds = find(w);
if ~isempty(nz_inds)
    str = [lhs," = ",num2str(w(nz_inds(1)),'%+.5f'), tags_pde{nz_inds(1)}];
    for k=2:length(nz_inds)
        str = [str,num2str(w(nz_inds(k)),'%+.5f'),tags_pde{nz_inds(k)}];
    end
else
    str = '';
end
end


function [str] = renameEqnPde(str)
% Copyright 2022, All Rights Reserved
% Code by Mengyi Tang
% For Paper, "WeakIdent: Weak formulation for Identifying
% Differential Equation using Narrow-fit and Trimming"
% by Mengyi Tang, Wenjing Liao, Rachel Kuske and Sung Ha Kang


str = strrep(str, 'u^{1}_{t}', 'u_t');
str = strrep(str, 'u^{1,0}_{t}', 'u_t');
str = strrep(str, 'u^{0,1}_{t}', 'v_t');
str = strrep(str, '_{}','');
for i = 0:6
    str1 = ['u^{',num2str(i),'}'];
    if i == 0
        str2 = '1';
    elseif i == 1
        str2 = 'u';
    else
        str2 = ['(u^', num2str(i),')'];
    end
    str = strrep(str, str1, str2);
end


for i =  0:6
    for j = 0:6
        str1 = ['u^{',num2str(i),',',num2str(j),'}'];
        if i == 0
            str2_x = '';
        elseif i == 1
            str2_x = 'u';
        elseif i >1
            str2_x = ['u^', num2str(i)];
        end
        
        if j == 0
            str2_y = '';
        elseif j == 1
            str2_y = 'v';
        elseif j >1
            str2_y = ['v^', num2str(j)];
        end
        
        str = strrep(str, str1, [str2_x,str2_y]);
    end
end

for i =  0:6
    for j = 0:6
        for k = 0:6
            str1 = ['u^{',num2str(i),',',num2str(j),',',num2str(k),  '}'];
            if i == 0
                str2_x = '';
            elseif i == 1
                str2_x = 'u';
            elseif i >1
                str2_x = ['u^', num2str(i)];
            end
            
            if j == 0
                str2_y = '';
            elseif j == 1
                str2_y = 'v';
            elseif j >1
                str2_y = ['v^', num2str(j)];
            end
            
            
            if k == 0
                str2_z = '';
            elseif k == 1
                str2_z = 'w';
            elseif k >1
                str2_z = ['w^', num2str(k)];
            end
            
            str = strrep(str, str1, [str2_x,str2_y,str2_z]);
        end
    end
    
    
end
end

function [str] = renameEqnOde(str)
% printODEEquationForPdeV1: to print the equations provided tags of features

% Copyright 2022, All Rights Reserved
% Code by Mengyi Tang
% For Paper, "WeakIdent: Weak formulation for Identifying
% Differential Equation using Narrow-fit and Trimming"
% by Mengyi Tang, Wenjing Liao, Rachel Kuske and Sung Ha Kang

str = strrep(str, 'u^{1,0}_{t}', '\dot{x}');
str = strrep(str, 'u^{0,1}_{t}', '\dot{y}');

str = strrep(str, 'u^{1,0,0}_{t}', '\dot{x}');
str = strrep(str, 'u^{0,1,0}_{t}', '\dot{y}');
str = strrep(str, 'u^{0,0,1}_{t}', '\dot{z}');

str = strrep(str, 'u^{1,0,0,0,0}_{t}', '\dot{x_1}');
str = strrep(str, 'u^{0,1,0,0,0}_{t}', '\dot{x_2}');
str = strrep(str, 'u^{0,0,1,0,0}_{t}', '\dot{x_3}');
str = strrep(str, 'u^{0,0,0,1,0}_{t}', '\dot{x_4}');
str = strrep(str, 'u^{0,0,0,0,1}_{t}', '\dot{x_5}');

str = strrep(str, '_{}','');

for i =  0:6
    for j = 0:6
        str1 = ['u^{',num2str(i),',',num2str(j),'}'];
        if i == 0
            str2_x = '';
        elseif i == 1
            str2_x = 'x';
        elseif i >1
            str2_x = ['x^', num2str(i)];
        end
        
        if j == 0
            str2_y = '';
        elseif j == 1
            str2_y = 'y';
        elseif j >1
            str2_y = ['y^', num2str(j)];
        end
        
        str = strrep(str, str1, [str2_x,str2_y]);
    end
end

for i =  0:6
    for j = 0:6
        for k = 0:6
            str1 = ['u^{',num2str(i),',',num2str(j),',',num2str(k),  '}'];
            if i == 0
                str2_x = '';
            elseif i == 1
                str2_x = 'x';
            elseif i >1
                str2_x = ['x^', num2str(i)];
            end
            
            if j == 0
                str2_y = '';
            elseif j == 1
                str2_y = 'y';
            elseif j >1
                str2_y = ['y^', num2str(j)];
            end
            
            
            if k == 0
                str2_z = '';
            elseif k == 1
                str2_z = 'z';
            elseif k >1
                str2_z = ['z^', num2str(k)];
            end
            
            str = strrep(str, str1, [str2_x,str2_y,str2_z]);
        end
    end
    
    
end

for i =  0:3
    for j = 0:3
        for k = 0:3
            for l = 0:3
                for m = 0:3
                    str1 = ['u^{',num2str(i),',',num2str(j),',',num2str(k),',',num2str(l),',',num2str(m),  '}'];
                    if i == 0
                        str2_x = '';
                    elseif i == 1
                        str2_x = 'x_1';
                    elseif i >1
                        str2_x = ['x_1^', num2str(i)];
                    end
                    
                    if j == 0
                        str2_y = '';
                    elseif j == 1
                        str2_y = 'x_2';
                    elseif j >1
                        str2_y = ['x_2^', num2str(j)];
                    end
                    
                    
                    if k == 0
                        str2_z = '';
                    elseif k == 1
                        str2_z = 'x_3';
                    elseif k >1
                        str2_z = ['x_3^', num2str(k)];
                    end
                    
                    if l == 0
                        str2_l = '';
                    elseif l == 1
                        str2_l = 'x_4';
                    elseif l >1
                        str2_l = ['x_4^', num2str(l)];
                    end
                    
                    if m == 0
                        str2_m = '';
                    elseif k == 1
                        str2_m = 'x_5';
                    elseif k >1
                        str2_m = ['x_5^', num2str(m)];
                    end
                    
                    str = strrep(str, str1, [str2_x,str2_y,str2_z,str2_l,str2_m]);
                end
            end
        end
    end
    
end
end
