clear;clc

load('problems5.mat'); 
lambdas = [0.1, 0.05, 0.01, 0.005];
xrandomseed = [5240,3196,3411,4718;3392,4023,4506,7395;1326,5278,967,6047;6218,5846,9937,7224;6321,9366,937,7136]; 

%% 调用NPDQNM求解LASSO
dimension = 2; 
datanum = 100; 
NPDQNM1data = zeros(datanum,dimension);
nrun = 0;

for numprob = 1:length(problems5)
    % 加载数据
    A = problems5{numprob}.A;
    b = problems5{numprob}.b;
    [~, n] = size(A);

    % 循环所有初始点
    for numpoint = 1:5
        % 生成初始点
        if numpoint == 1
            x0 = zeros(n, 1); % 零向量
        else
            seed = xrandomseed(numprob,numpoint-1);
            rng(seed); % 固定随机种子
            x0 = randn(n, 1); % 随机向量
        end

        % 循环所有lambda
        for numlam = 1:length(lambdas)
            lambda = lambdas(numlam);

            % 配置算法选项 
            opts = struct();
            opts.M = 5;
            opts.gamma = 1e-4;
            opts.sigma = 0.75;
            opts.lambdayi = 0.95;
            opts.gamma1 = 1e-10;
            opts.gamma2 = 1e10;
            opts.tol = 1e-6;
            opts.max_iter = 10000;
            opts.lambda = lambda;

            % 调用目标函数光滑项及其梯度
            fun = @(x) LASSOf(x, A, b);
            gradf = @(x) LASSOgradf(x, A, b);
            % 定义非光滑项
            g = @(x) lambda*norm(x, 1);

            nrun = nrun + 1; 
            fprintf('\n 第 %d 个数据集：\n',nrun)

            % 调用算法求解LASSO问题
            tic;
            [iter, x_opt, F_opt, F_history] = NPDQNM1(x0, fun, gradf, g, opts);
            ttime = toc;

            % 储存当前数据
            NPDQNM1data(nrun, :) = [iter, ttime];
        end
    end
end

save NPDQNM1data.mat NPDQNM1data

%% 调用mless_BFGS求解LASSO
dimension = 2; 
datanum = 100; 
mless_BFGSdata = zeros(datanum,dimension);
nrun = 0;

for numprob = 1:length(problems5)
    % 加载数据
    A = problems5{numprob}.A;
    b = problems5{numprob}.b;
    [~, n] = size(A);

    % 循环所有初始点
    for numpoint = 1:5
        % 生成初始点
        if numpoint == 1
            x0 = zeros(n, 1); % 零向量
        else
            seed = xrandomseed(numprob,numpoint-1);
            rng(seed); % 固定随机种子
            x0 = randn(n, 1); % 随机向量
        end

        % 循环所有lambda
        for numlam = 1:length(lambdas)
            lambda = lambdas(numlam);

            % 配置算法选项 
            opts = struct();
            opts.nu1 = 1e-8;
            opts.thetak = 0.1;
            opts.delta = 1e-3;
            opts.beta = 0.5;
            opts.tol = 1e-6;                
            opts.max_iter = 10000;           
            opts.lambda = lambda;

            % 调用目标函数光滑项及其梯度 
            fun = @(x) LASSOf(x, A, b);
            gradf = @(x) LASSOgradf(x, A, b);
            % 定义非光滑项
            g = @(x) lambda*norm(x, 1);

            nrun = nrun + 1;  
            fprintf('\n 第 %d 个数据集：\n',nrun)

            % 调用算法求解LASSO问题
            tic;
            [iter, x_opt, F_opt] = mless_BFGS1(x0, fun, gradf, g, opts);
            ttime = toc;

            % 储存当前数据
            mless_BFGSdata(nrun, :) = [iter, ttime];
        end
    end
end

save mless_BFGSdata.mat mless_BFGSdata

%% 调用mless_SR1求解LASSO
dimension = 2; 
datanum = 100;
mless_SR1data = zeros(datanum,dimension);
nrun = 0;

for numprob = 1:length(problems5)
    % 加载数据
    A = problems5{numprob}.A;
    b = problems5{numprob}.b;
    [~, n] = size(A);

    % 循环所有初始点
    for numpoint = 1:5
        % 生成初始点
        if numpoint == 1
            x0 = zeros(n, 1); % 零向量
        else
            seed = xrandomseed(numprob,numpoint-1);
            rng(seed); % 固定随机种子
            x0 = randn(n, 1); % 随机向量
        end

        % 循环所有lambda
        for numlam = 1:length(lambdas)
            lambda = lambdas(numlam);

            % 配置算法选项
            opts = struct();
            opts.nu1 = 1e-8;
            opts.thetak = 0.1;
            opts.delta = 1e-3;
            opts.beta = 0.5;
            opts.tol = 1e-6;                 
            opts.max_iter = 10000;           
            opts.lambda = lambda;

            % 调用目标函数光滑项及其梯度 
            fun = @(x) LASSOf(x, A, b);
            gradf = @(x) LASSOgradf(x, A, b);
            % 定义非光滑项
            g = @(x) lambda*norm(x, 1);

            nrun = nrun + 1;  
            fprintf('\n 第 %d 个数据集：\n',nrun)

            % 调用算法求解LASSO问题
            tic;
            [iter, x_opt, F_opt] = mless_SR11(x0, fun, gradf, g, opts);
            ttime = toc;

            % 储存当前数据
            mless_SR1data(nrun, :) = [iter, ttime];
        end
    end
end

save mless_SR1data.mat mless_SR1data

%% 调用zeroSR1算法求解
% 添加路径
addpath('D:\MATLABcode\Proximal\Proximal Methods5\LASSO\zeroSR1-master'); 
dimension = 2;
datanum = 100; 
zeroSR1data = zeros(datanum,dimension);
nrun = 0;

for numprob = 1:length(problems5)
    % 加载数据
    A = problems5{numprob}.A;
    b = problems5{numprob}.b;
    [~, n] = size(A);

    % 循环所有初始点
    for numpoint = 1:5
        % 生成初始点
        if numpoint == 1
            x0 = zeros(n, 1); % 零向量
        else
            seed = xrandomseed(numprob,numpoint-1);
            rng(seed); % 固定随机种子
            x0 = randn(n, 1); % 随机向量
        end

        % 循环所有lambda
        for numlam = 1:length(lambdas)
            lambda = lambdas(numlam);

            % 配置算法选项 
            opts = struct();
            opts.x0 = x0;                        
            opt.grad_tol = 1e-6;            
            opts.tol = 1e-9;                
            opts.nmax = 10000;               
            opts.verbose = 1;                

            % 调用目标函数光滑项及其梯度 
            fcn = @(x) LASSO_loss(x, A, b);
            grad = @(x) LASSO_gradient(x, A, b);

            % 定义非光滑项
            h = @(x) lambda*norm(x, 1);

            % 定义非光滑函数的prox算子，使用zeroSR1所需的特殊prox_rank1_l1格式
            prox = @(x0,d,v) prox_rank1_l1(x0, d, v, lambda);

            nrun = nrun + 1;  
            fprintf('\n 第 %d 个数据集：\n',nrun)

            % 调用算法求解LASSO问题
            tic;
            [xk, nit, errStruct] = zeroSR1(fcn, grad, h, prox, opts);
            ttime = toc;
            iter = nit;

            % 储存当前数据
            zeroSR1data(nrun, :) = [iter, ttime];
        end
    end
end

save zeroSR1data.mat zeroSR1data


%% 自建LASSO光滑部分函数
function fx = LASSOf(w,A,b)
r = A*w - b;
fx = 0.5*(r'*r);
end

function gradfx = LASSOgradf(w,A,b)
r = A*w - b;
gradfx = A' * r;  % 计算f对应梯度
end

%% ==== 辅助函数：逻辑回归损失和梯度 ====
function loss = LASSO_loss(w, A, b)
r = A*w - b;
loss = 0.5*(r'*r);
end

function grad = LASSO_gradient(w, A, b)
r = A*w - b;
grad = A' * r;
end
