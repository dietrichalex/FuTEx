clear; clc; close all;
mu = [0; 0];
N = 2000;
n = length(mu); 
plot_limit = 35;
ax_limits = [-plot_limit, plot_limit, -plot_limit, plot_limit];

figure('Name', 'Variation der Kovarianzmatrix', 'Position', [100, 100, 700, 600]);
h_scatter = scatter(0, 0, 10, 'filled', 'MarkerFaceAlpha', 0.3);
axis equal;
grid on;
xlabel('X1');
ylabel('X2');


%% Variation of Sigma_11
Sigma_22 = 100;
Sigma_12 = 0;
var_range = linspace(1, 100, 80);

for s11 = var_range
    Sigma = [s11, Sigma_12; 
             Sigma_12, Sigma_22];
    
    [V, D] = eig(Sigma);
    A = V * sqrt(D);
    Z = randn(n, N);
    X = A * Z;
   
    set(h_scatter, 'XData', X(1,:), 'YData', X(2,:));
    axis(ax_limits);
    title(sprintf('Variation von \\Sigma_{11}\n \\Sigma = [%.1f, %.1f; %.1f, %.1f]', ...
          Sigma(1,1), Sigma(1,2), Sigma(2,1), Sigma(2,2)));
    pause(0.05);
end
pause(1.0);


%% Variation of Sigma_22
Sigma_11 = 100;
Sigma_12 = 0;

for s22 = var_range
    Sigma = [Sigma_11, Sigma_12; 
             Sigma_12, s22];
   
    [V, D] = eig(Sigma);
    A = V * sqrt(D);
    Z = randn(n, N);
    X = A * Z;
    
    set(h_scatter, 'XData', X(1,:), 'YData', X(2,:));
    axis(ax_limits);
    title(sprintf('Variation von \\Sigma_{22}\n \\Sigma = [%.1f, %.1f; %.1f, %.1f]', ...
          Sigma(1,1), Sigma(1,2), Sigma(2,1), Sigma(2,2)));
    pause(0.05);
end
pause(1.0);


%% Variation of Sigma_12
Sigma_11 = 100;
Sigma_22 = 100;
cov_range = linspace(-99.9, 99.9, 150); 
cov_range = [cov_range, fliplr(cov_range)]; 

for s12 = cov_range
    Sigma = [Sigma_11, s12; 
             s12, Sigma_22];
    
    [V, D] = eig(Sigma);
    A = V * sqrt(D);
    Z = randn(n, N);
    X = A * Z;
    
    set(h_scatter, 'XData', X(1,:), 'YData', X(2,:));
    axis(ax_limits);
    title(sprintf('Variation von \\Sigma_{12}\n \\Sigma = [%.1f, %.1f; %.1f, %.1f]', ...
          Sigma(1,1), Sigma(1,2), Sigma(2,1), Sigma(2,2)));
    pause(0.05);
end