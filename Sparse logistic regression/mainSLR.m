clear;clc

% 数据集
set = {'w1at.mat','w2a.mat','w3a.mat','w4a.mat','w8at.mat','ijcnn1t.mat'};
nrun = 10; 
num = length(set);
dataset = zeros(num,2); 

lambda = 1e-3;       % 正则化参数

% 定义非光滑项
g = @(x) lambda*norm(x, 1);

%% 调用NPDQNM算法求解
for ii = 1:num
    fprintf('第 %d 个数据集：\n',ii);
    % 加载数据集
    data = load(set{ii});
    A = data.A;  b = data.b;

    [m,n] = size(A);
    x0 = zeros(n,1);

    % 配置算法选项 
    opts = struct();
    opts.M = 5;                
    opts.gamma = 1e-4;                  
    opts.sigma = 0.75;                
    opts.lambdayi = 0.95;   
    opts.gamma1 = 1e-10;
    opts.gamma2 = 1e10;
    opts.tol = 1e-6;
    opts.ftol = 1e-9;
    opts.max_iter = 10000;           
    opts.lambda = lambda; 

    % 调用目标函数光滑项及其梯度 
    fun = @(x) logistic_loss(x, A, b);
    gradf = @(x) logistic_gradient(x, A, b);

    dataset1 = zeros(nrun,2); 

    for jj = 1:nrun
        tic;
        [iter, x_opt, F_opt, F_history] = NPDQNM1(x0, fun, gradf, g, opts);
        ttime = toc;

        dataset1(jj,:) = [iter,ttime];
    end

    dataset(ii, 1) = mean(dataset1(:, 1)); 
    dataset(ii, 2) = mean(dataset1(:, 2)); 
end

save NPDQNM1.mat dataset

% %% 调用mless_SR1算法求解
% for ii = 1:num
%     fprintf('第 %d 个数据集：\n',ii);
%     % 加载数据集
%     data = load(set{ii});
%     A = data.A;  b = data.b;
% 
%     [m,n] = size(A);
%     x0 = zeros(n,1);
% 
%     % 配置算法选项
%     opts = struct();
%     opts.nu1 = 1e-8;                
%     opts.thetak = 0.1;                  
%     opts.delta = 1e-3;                
%     opts.beta = 0.5;   
%     opts.tol = 1e-6;                 
%     opts.ftol = 1e-9;
%     opts.max_iter = 10000;           
%     opts.lambda = lambda; 
% 
%     % 调用目标函数光滑项及其梯度
%     fun = @(x) logistic_loss(x, A, b);
%     gradf = @(x) logistic_gradient(x, A, b);
% 
%     dataset1 = zeros(nrun,2);
% 
%     for jj = 1:nrun
%         tic;
%         [iter, x_opt, F_opt] = mless_SR11(x0, fun, gradf, g, opts);
%         ttime = toc;
% 
%         dataset1(jj,:) = [iter,ttime];
%     end
% 
%     dataset(ii, 1) = mean(dataset1(:, 1)); 
%     dataset(ii, 2) = mean(dataset1(:, 2)); 
% end
% 
% save mless_SR1.mat dataset
% 
% %% 调用mless_BFGS算法求解
% for ii = 1:num
%     fprintf('第 %d 个数据集：\n',ii);
%     % 加载数据集
%     data = load(set{ii});
%     A = data.A;  b = data.b;
% 
%     [m,n] = size(A);
%     x0 = zeros(n,1);
% 
%     % 配置算法选项
%     opts = struct();
%     opts.nu1 = 1e-8;                
%     opts.thetak = 0.1;                  
%     opts.delta = 1e-3;                
%     opts.beta = 0.5;   
%     opts.tol = 1e-6;                 
%     opts.ftol = 1e-9;
%     opts.max_iter = 10000;           
%     opts.lambda = lambda; 
% 
%     % 调用目标函数光滑项及其梯度
%     fun = @(x) logistic_loss(x, A, b);
%     gradf = @(x) logistic_gradient(x, A, b);
% 
%     dataset1 = zeros(nrun,2); 
% 
%     for jj = 1:nrun
%         tic;
%         [iter, x_opt, F_opt] = mless_BFGS1(x0, fun, gradf, g, opts);
%         ttime = toc;
% 
%         dataset1(jj,:) = [iter,ttime];
%     end
% 
%     dataset(ii, 1) = mean(dataset1(:, 1));
%     dataset(ii, 2) = mean(dataset1(:, 2));
% end
% 
% save mless_BFGS.mat dataset


%% ==== 辅助函数：逻辑回归损失和梯度 ====
function loss = logistic_loss(w, A, b)
z = b.*(A*w);
loss = sum(log(1 + exp(-z))) / length(b);
end

function grad = logistic_gradient(w, A, b)
z = b.*(A*w);
p = 1 ./ (1 + exp(z));
grad = A'*(-b.*p) / length(b);
end
