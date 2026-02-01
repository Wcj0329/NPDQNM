clear;clc
randomseed = [4857,990276,814724,363434,906159];
i = [8,9,10,11,12];
problems5 = cell(5,1);
idx = 1;

for j = 1:5 
    rng(randomseed(j));
    jj = i(j);
    m = 2^(jj+1);
    n = 2^(jj);
    
    A_raw = randn(m, n);
    b = randn(m, 1);
    
    % ¹éÒ»»¯
    A = zeros(size(A_raw));
    for col = 1:n
        col_norm = norm(A_raw(:, col));
        if col_norm > eps
            A(:, col) = A_raw(:, col) / col_norm;
        else
            A(:, col) = A_raw(:, col); 
        end
    end
    
    problems5{idx} = struct('A', A, 'b', b);
    idx = idx + 1;
end

save('problems5.mat', 'problems5');