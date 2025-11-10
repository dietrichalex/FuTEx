clear;
clc;
close all;

Y_data = load('samples.txt');
[T, N] = size(Y_data);
t = (1:N)';
A = [ones(N, 1), t, t.^2, t.^3];

figure;
hold on;
grid on;

min_val = min(Y_data(:));
max_val = max(Y_data(:));
buffer = (max_val - min_val) * 0.1;
ylim([min_val - buffer, max_val + buffer]);
xlim([1, N]);

h_data = plot(t, Y_data(1,:)', 'bo', 'MarkerSize', 4);
h_fit  = plot(t, zeros(N, 1), 'r-', 'LineWidth', 2); 


xlabel('Stützstelle (t)');
ylabel('Spannungssignal (y)');
h_title = title(sprintf('Polynom-Approximation (Grad 3) - Zeitschritt T = 1 / %d', T));
legend('Abgetastete Daten', 'Polynom-Fit', 'Location', 'SouthEast');

for k = 1:T
    y = Y_data(k, :)';

    %c = (A^T * A)^(-1) * (A^T * y) <- Normalengleichung
    c = A \ y;
    y_fit = A * c;
    
    set(h_data, 'YData', y);
    set(h_fit, 'YData', y_fit); 
    set(h_title, 'String', sprintf('Polynom-Approximation (Grad 3) - Zeitschritt T = %d / %d', k, T));
    pause(0.01);
end