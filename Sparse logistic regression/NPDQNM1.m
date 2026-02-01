function [iter, x_opts, F_opts, F_history] = NPDQNM1(x0, f, gradfun, g, opts) 
% 设置参数
params = struct('M', 5, 'gamma', 1e-4, 'sigma', 0.75, 'lambdayi', 0.95, ...
    'gamma1', 1e-10, 'gamma2', 1e10, 'max_iter', 10000, 'tol', 1e-6,'ftol',1e-9, 'lambda', 1e-3);

% 解析可选参数
if nargin > 4
    if isstruct(opts)
        fields = fieldnames(opts);
        for i = 1:length(fields)
            if isfield(params, fields{i})
                params.(fields{i}) = opts.(fields{i});
            else
                warning('未知参数: %s', fields{i});
            end
        end
    end
end

% 初始化参数
k = 0;
n = length(x0);
I = ones(n,1);
D = I;
fval = f(x0);
gradf = gradfun(x0);
gval = g(x0);
Fval = fval + gval;
F_history = Fval;  

fprintf('  iter     normd        Fval\n');

while k<params.max_iter
    %% 计算残差
    x_plus = prox(x0,D,gradf,params.lambda);
    dk = x_plus - x0;
    normd = norm(dk,'inf');
    
    fprintf('%3.0d      %3.2e         %3.4f\n', k, normd, Fval);
    
    %% 终止条件
    if normd < params.tol
        break;
    end
    
    
    %% 更新步长alpha
    wk = min(length(F_history), params.M+1); 
    weighted_f = zeros(wk,1);
    
    % 计算权重函数
    for j = 1:wk
        jj = j - 1;
        fjj = F_history(end-jj);
        exponent = sign(fjj)*jj;      
        weighted_f(j) = (params.lambdayi^exponent) * fjj;
    end
    
    F_pk = max(weighted_f);       
    
    xkd = x0 + dk;
    gvalxkd = g(xkd);
    Delta = gradf'*dk + gvalxkd - gval;
    
    % 寻找步长
    alpha = 1;
    while 1
        x1 = x0 + alpha*dk;
        fval1 = f(x1);
        gval1 = g(x1);
        Fval1 = fval1 + gval1;
        
        % 检查Armijo条件
        if Fval1 <= F_pk + params.gamma*alpha*Delta
            break;
        end
        
        % 回溯步长
        alpha = alpha * params.sigma;
        
        % % 防止无限循环
        % if alpha < 1e-15
        %     warning('步长过小，线搜索终止');
        %     break;
        % end
    end
    
    
    %% 终止条件(另一个终止条件)
    df  = abs(Fval1 - Fval)/max(1,abs(Fval));
    if df < params.ftol
        break;
    end
    
    %% 更新下一个迭代点
    % x1 = x0 + alpha*dk;
    
    %% 更新历史函数值（维护窗口大小M+1）
%     if length(F_history) >= params.M+1
%         F_history = [F_history(2:end), Fval1];
%     else
%         F_history = [F_history, Fval1];
%     end
    
    F_history = [F_history, Fval1];
    if length(F_history) > params.M
        F_history = F_history(end-params.M:end);
    end
    
    %% 进入下一次循环
    gradf1 = gradfun(x1);
    s = x1 - x0;
    y = gradf1 - gradf;
    sTs = s'*s; sTy = s'*y; yTy = y'*y;
    theta = (sTs)/(sTy) - sqrt(((sTs)/(sTy))^2 - (sTs)/(yTy));
    
    %% 更新逆Hessian对角近似
    styy = (s-theta*y)'*y; sty = s-theta*y;
    if styy ~= 0
        D1 = theta*I + ((sty.^2)/styy);
        D = min(max(D1,params.gamma1),params.gamma2);
    else
        D = I;
    end
    
    k = k + 1;
    x0 = x1;
    fval = fval1;
    gval = gval1;
    gradf = gradf1;
    Fval = Fval1;
end
iter = k;
x_opts = x0;
F_opts = Fval;