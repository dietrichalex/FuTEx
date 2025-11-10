clear;
clc;
close all;

X = load('norm2D.txt');
X = X';
mu = mean(X);
Sigma = cov(X);

std_devs = sqrt(diag(Sigma));
range_multiplier = 3.5;

x1_range = linspace(mu(1) - range_multiplier * std_devs(1), mu(1) + range_multiplier * std_devs(1), 150);
x2_range = linspace(mu(2) - range_multiplier * std_devs(2), mu(2) + range_multiplier * std_devs(2), 150);

[X1_mesh, X2_mesh] = meshgrid(x1_range, x2_range);

GridPoints = [X1_mesh(:), X2_mesh(:)];

P_values = mvnpdf(GridPoints, mu, Sigma);
P_mesh = reshape(P_values, size(X1_mesh));

figure;
surf(X1_mesh, X2_mesh, P_mesh);
shading interp;
colorbar;
title('PDF');
xlabel('X1');
ylabel('X2');
zlabel('f(x1, x2)');
axis tight;

figure;
scatter(X(:,1), X(:,2), 15, 'filled', 'MarkerFaceAlpha', 0.4);
hold on;

[M, c] = contour(X1_mesh, X2_mesh, P_mesh, 'LineWidth', 1.5);
c.LevelList = round(c.LevelList, 3);
clabel(M, c);

plot(mu(1), mu(2), 'rx', 'MarkerSize', 14, 'LineWidth', 2);

hold off;
axis equal;
grid on;
title('Punktwolke');
xlabel('X1');
ylabel('X2');
