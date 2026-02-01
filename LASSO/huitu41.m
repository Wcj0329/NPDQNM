clear,clc
S1=load('ceshi46.mat');BP1 = struct2cell(S1);Mymat11 = cell2mat(BP1);
[hhh,~]=size(Mymat11);

% (I) Iterations 
for dd=1:4
    ii=1;
    k1 = Mymat11(:, ii);        
    k2 = Mymat11(:, ii+2);      
    k3 = Mymat11(:, ii+4);      
    k4 = Mymat11(:, ii+6);      
    
    R = zeros(hhh, 4);
    
    for q = 1:hhh
        k_values = [k1(q), k2(q), k3(q), k4(q)];
        a = min(k_values);
        R(q, :) = k_values / a;  
    end
    
    R(isnan(R)) = 10000;
    performance_ratios = R;
    
    n_solvers = 4;
    tau_values = linspace(1, 100, 1000);  
    rho = zeros(length(tau_values), n_solvers);
    
    for s = 1:n_solvers
        for i = 1:length(tau_values)
            rho(i, s) = sum(performance_ratios(:, s) <= tau_values(i)) / hhh;
        end
    end
    
    figure(1);
    colors = [0,0.78,0.55; 0,0.4470,0.7410; 0.93,0.69,0.13; 0.49,0.18,0.56];
    line_styles = {'-', '-', '-', '-'};
    solver_names = {'zeroSR1', 'Imless-SR1', 'Imless-BFGS', 'NPDQNM'};
    markers = {'o', 'p', 's', 'd'};  
    marker_sizes = [6, 6, 6, 6];
    
    hold on;
    for s = 1:n_solvers
        plot(tau_values, rho(:, s), line_styles{s}, ...
            'Color', colors(s, :), 'LineWidth', 1.5, ...
            'Marker', markers{s}, 'MarkerSize', marker_sizes(s), ...
            'MarkerEdgeColor', colors(s, :), ...
            'MarkerFaceColor', colors(s, :));  
    end
    
    axis([0.95 3.05 -0.04 1.04]);
    xticks(1:0.5:3);
    yticks(0:0.2:1);
    xlabel('\tau', 'Interpreter', 'tex');
    ylabel('\rho(\tau)', 'Interpreter', 'tex');
    legend(solver_names, 'Interpreter', 'latex', 'Location', 'southeast');
    title('Number of iterations', 'FontName', 'Times New Roman', 'FontSize', 13);
    hold off;
    ax = gca;
    set(ax, 'Box', 'on');
end

% (II) CPU time 
for dd=1:4
    ii=2;
    k1 = Mymat11(:, ii);        
    k2 = Mymat11(:, ii+2);      
    k3 = Mymat11(:, ii+4);      
    k4 = Mymat11(:, ii+6);      
    
    R = zeros(hhh, 4);
    
    for q = 1:hhh
        k_values = [k1(q), k2(q), k3(q), k4(q)];
        a = min(k_values);
        R(q, :) = k_values / a;
    end
    
    R(isnan(R)) = 10000;
    performance_ratios = R;
    
    n_solvers = 4;
    tau_values = linspace(1, 100, 1000);
    rho = zeros(length(tau_values), n_solvers);
    
    for s = 1:n_solvers
        for i = 1:length(tau_values)
            rho(i, s) = sum(performance_ratios(:, s) <= tau_values(i)) / hhh;
        end
    end
    
    figure(3);
    colors = [0,0.78,0.55; 0,0.4470,0.7410; 0.93,0.69,0.13; 0.49,0.18,0.56];
    line_styles = {'-', '-', '-', '-'};
    solver_names = {'zeroSR1', 'Imless-SR1', 'Imless-BFGS', 'NPDQNM'};
    markers = {'o', 'p', 's', 'd'}; 
    marker_sizes = [6, 6, 6, 6];
    
    hold on;
    for s = 1:n_solvers
        plot(tau_values, rho(:, s), line_styles{s}, ...
            'Color', colors(s, :), 'LineWidth', 1.5, ...
            'Marker', markers{s}, 'MarkerSize', marker_sizes(s), ...
            'MarkerEdgeColor', colors(s, :), ...
            'MarkerFaceColor', colors(s, :));
    end
    
    axis([0.95 3.05 -0.04 1.04]);
    xticks(1:0.5:3);
    yticks(0:0.2:1);
    xlabel('\tau', 'Interpreter', 'tex');
    ylabel('\rho(\tau)', 'Interpreter', 'tex');
    legend(solver_names, 'Interpreter', 'latex', 'Location', 'southeast');
    title('CPU time', 'FontName', 'Times New Roman', 'FontSize', 13);
    hold off;
    ax = gca;
    set(ax, 'Box', 'on');
end