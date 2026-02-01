clear;clc

%% 添加路径
addpath('D:\MATLABcode\Proximal\Proximal Methods4\Sparse logistic regression\zeroSR1-master'); 

%% 数据集
set = {'w1at.mat','w2a.mat','w3a.mat','w4a.mat','w8at.mat','ijcnn1t.mat'};

%% 预设数组
nrun = 10; 
num = length(set);
dataset = zeros(num,2); 

%% 定义问题参数 (文献4.2节设置)
lambda = 1e-3;       % 正则化参数

%% 定义非光滑项
h = @(x) lambda*norm(x, 1);

%% 定义非光滑函数的prox算子，使用zeroSR1所需的特殊prox_rank1_l1格式
prox = @(x0,d,v) prox_rank1_l1(x0, d, v, lambda);

%% 调用zeroSR1_5算法求解
for ii = 1:num
    fprintf('第 %d 个数据集：\n',ii);
    % 加载数据集
    data = load(set{ii});
    A = data.A;  b = data.b;

    [m,n] = size(A);
    x0 = zeros(n,1);

    % 配置算法选项 
    opts = struct();
    opts.x0 = x0;                   
    opts.tol = 1e-9;                
    opts.grad_tol=1e-6;              
    opts.nmax = 10000;              
    opts.verbose = 1;                

    % 调用目标函数光滑项及其梯度 
    fcn = @(x) logistic_loss(x, A, b);
    grad = @(x) logistic_gradient(x, A, b);

    dataset1 = zeros(nrun,2); 

    for jj = 1:nrun
        tic;
        [xk, nit, errStruct] = zeroSR1(fcn, grad, h, prox, opts);
        ttime = toc;
        iter = nit;

        dataset1(jj,:) = [iter,ttime];
    end

    dataset(ii, 1) = mean(dataset1(:, 1));
    dataset(ii, 2) = mean(dataset1(:, 2));
end

save zeroSR1.mat dataset

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

