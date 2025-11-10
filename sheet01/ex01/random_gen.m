%% Verfahren
% mu: Ziel-Mittelwert
mu = [10; 
      5];

% Ziel-Kovarianzmatrix
Sigma = [4, 3; 
         3, 9];

n = length(mu);
N = 10000;


% cholesky mit sigma = A * A^T  oder sigma = V * (D^(1/2))
A = chol(Sigma, 'lower');

% Z als N(0, I) Verteilung.
Z = randn(n, N);


% transformation
X = A * Z + mu;

%% Plots

X_T = X';

mu_t = mean(X_T);
Sigma_t = cov(X_T);

figure;
scatter(X(1,:), X(2,:), 10, 'filled', 'MarkerFaceAlpha', 0.2);
title('N(mu, Sigma)');
xlabel('X1');
ylabel('X2');
axis equal;
grid on;
hold on;

plot(mu(1), mu(2), 'rx', 'MarkerSize', 14, 'LineWidth', 2);