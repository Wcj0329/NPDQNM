function [iter, x_opt, F_opt] = mless_BFGS1(x0, f, gradfun, g, opts)
% 参数
params = struct('nu1', 1e-8, ...% 公式(33)所需参数
    'thetak', 0.1, ... % 公式(31)所需参数
    'delta', 1e-3, 'beta', 0.5, ...% 线搜索参数
    'max_iter', 10000, 'tol', 1e-6,'lambda', 1e-3);

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

k = 0;
n = length(x0);
I = ones(n,1);
favl = f(x0);
gradf = gradfun(x0);
gval = g(x0);
Fval = favl + gval;

fprintf('\n  iter     normd        Fval');

while k<params.max_iter
    %% Step1: 求解子问题x_plus
    if k == 0
        u1 = x0 - I.*gradf;
        x_plus = sign(u1).*max(abs(u1)-params.lambda,0);
        % x_plus = prox(x0,I,gradf,params.lambda);

    else
        %% 利用V-FISTA非精确求解子问题x_plus
        lambdakw = 1 - Phik + (gammak+Phik*(sTs/sTz))*(zTz/sTz);       % 计算lambdak弯
        L = lambdakw^2 - 4*gammak*(Phik*(zTz/sTz)+(1-Phik)*(sTz/sTs));

        lambdak1 = (1/2) * (lambdakw+sqrt(L)); % 计算lambdak正
        lambdak2 = (1/2) * (lambdakw-sqrt(L)); % 计算lambdak负

        Kappak = lambdak1 / lambdak2; % 子问题(61)的条件数
        w = (sqrt(Kappak)-1)/(sqrt(Kappak)+1);

        % V-FISTA内部迭代
        wj = x0;
        wj_1 = wj;

        while 1
            wj_bar = wj + w * (wj-wj_1); wx = wj_bar-x0;

            Bwx = wx - (s'*wx/sTs)*s + gammak*(z'*wx/sTz)*z + Phik*(nu'*wx)*nu; % 计算B*(wj_bar-x0)
            grad_q = gradf + Bwx;
            u2 = wj_bar - (1/lambdak1)*grad_q;
            w_new = sign(u2).*max(abs(u2)-params.lambda/lambdak1,0);
            % w_new = prox(wj_bar,1/lambdak1,grad_q,params.lambda);

            % 内部迭代判断条件对应公式(31)
            ww = wj_bar-w_new;
            Bww = ww - (s'*ww/sTs)*s + gammak*(z'*ww/sTz)*z + Phik*(nu'*ww)*nu; % 计算B*(wj_bar-w_new)
            rj = lambdak1*(wj_bar-w_new) - Bww;
            dj = w_new - x0;

            dTd = dj'*dj; dTs = dj'*s; dTz = dj'*z; dTnu = dj'*nu; rTr = rj'*rj; rTs = rj'*s; rTz = rj'*z; rTwu = rj'*wu;

            rHr = rTr - (rTz^2)/zTz + (1/gammak)*(rTs^2)/sTz +  PhikH*(rTwu^2);
            dBd = dTd - (dTs^2)/sTs + gammak*(dTz^2)/sTz + Phik*(dTnu^2);
            rHr_norm = sqrt(rHr);  dBd_norm = sqrt(dBd);

            if rHr_norm <= (1-params.thetak)*dBd_norm % 判断公式(31)
                x_plus = w_new;
                break
            end

            wj_1 = wj;
            wj = w_new;
        end
    end

    %% 计算方向dk
    dk = x_plus - x0;

    %% 终止条件
    normd = norm(dk,'inf');
    fprintf('\n %3.0d      %3.2e         %3.4f',k,normd,Fval);

    if normd <= params.tol
        break;
    end

    %% 更新步长alphak
    alpha = 1;
    gvalxplus = g(x_plus);
    Delta = gradf'*dk + gvalxplus - gval;

    % 寻找步长初始化
    while 1
        x1 = x0 + alpha*dk;
        fval1 = f(x1);
        gval1 = g(x1);
        Fval1 = fval1 + gval1;

        % 检查Armijo条件
        if Fval1 <= Fval + params.delta*alpha*Delta
            break
        else
            alpha = alpha * params.beta;
        end

        % % 防止无限循环
        % if alpha < 1e-15
        %     warning('步长过小');
        %     % break;
        % end
    end
    alphak = alpha;

    %% 更新下一个迭代点
    x1 = x0 + alphak*dk;

    %% 计算nuk、gammak、Phik、Bk(14)和Hk(17)
    gradf1 = gradfun(x1);
    s = x1 - x0;
    y = gradf1 - gradf;

    sTy = s'*y;  sTs = s'*s;
    nu1ns = params.nu1 * (norm(s)^2);

    % 计算nuk(33)
    if sTy >= nu1ns
        nuk = 0;
    else
        max1 = -sTy/sTs;
        nuk = max(max1,0) + nu1;
    end

    z = y + nuk*s;

    sTz = s'*z; zTz = z'*z;

    % 计算gammak(34)
    gammak = sTz / zTz;

    % BFGS公式对应Phi值
    Phik = 0;  PhikH = 1;

    % 计算Bk参数nuk
    nu = sqrt(sTs)*(z/sTz - s/sTs);
    % B = I - (s*s')/sTs + gammak*(z*z')/sTz + Phik*(nu*nu');

    % 计算Hk参数wuk
    wu = sqrt(zTz)*(s/sTz - z/zTz);
    % H = I - (z*z')/zTz + (1/gammak)*(s*s')/sTz + PhikH*(wu*wu');

    %% 进入下一次循环
    k = k + 1;
    x0 = x1;
    fval = fval1;
    gval = gval1;
    gradf = gradf1;
    Fval = Fval1;
end
iter = k;
x_opt = x0;
F_opt = Fval;

