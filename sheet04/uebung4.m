function uebung2
%close all;

% history size: the number of measurements, estimations,... to store
global HIST_SIZE;
HIST_SIZE = 100;

% constant velocity of the true target
V_CONST = 100;

% the time between two measurements
T = 0.02;

% measurement noise (given)
sigma_y = 1;
sigma_z = 1;

% ----------------- histories & state variables ---------------------------
x_est       = [];  % state estimate [y; z; vy; vz]
X_Hist      = [];  % true state history
X_est_Hist  = [];  % estimation history
Z_Hist      = [];  % measurement history

% NEES/NIS histories
NEES_Hist = nan(1, HIST_SIZE);
NIS_Hist  = nan(1, HIST_SIZE);
k_step    = 0;                  % time-step counter for circular buffer

% ----------------- α-β tracker model matrices ---------------------------
% constant-velocity motion model for [y; z; vy; vz]
F = [1 0 T 0;
     0 1 0 T;
     0 0 1 0;
     0 0 0 1];

% only position [y; z]
H = [1 0 0 0;
     0 1 0 0];

% ----------------- α-β gains from λ -------------------------------------
v_max = 200;
sigma_v = v_max / 10;

% λ = σ_v^2 T^2 / σ_w^2   (lecture)
lambda_y = (sigma_v^2 * T^2) / (sigma_y^2);
lambda_z = (sigma_v^2 * T^2) / (sigma_z^2);

% α, β formulas -> lecture (per axis)
alpha_y = -1/8 * (lambda_y^2 + 8*lambda_y ...
           - (lambda_y + 4)*sqrt(lambda_y^2 + 8*lambda_y));
beta_y  =  1/4 * (lambda_y^2 + 4*lambda_y ...
           - lambda_y*sqrt(lambda_y^2 + 8*lambda_y));

alpha_z = -1/8 * (lambda_z^2 + 8*lambda_z ...
           - (lambda_z + 4)*sqrt(lambda_z^2 + 8*lambda_z));
beta_z  =  1/4 * (lambda_z^2 + 4*lambda_z ...
           - lambda_z*sqrt(lambda_z^2 + 8*lambda_z));

% Gain matrix K (4x2) for [y; z; vy; vz] <- innovation in [y; z]
K = [alpha_y     0;
     0        alpha_z;
     beta_y/T   0;
     0       beta_z/T];

% ----------------- covariance & consistency setup -----------------------
% Simple process/measurement noise models
Q = diag([0, 0, sigma_v^2, sigma_v^2]);    % process noise on velocities
R = diag([sigma_y^2, sigma_z^2]);          % measurement noise

P = 1e3 * eye(4);    % initial state covariance (large uncertainty)

nx = 4;              % state dimension
nz = 2;              % measurement dimension

% 95% chi-square thresholds (hard-coded values)
P95_NEES = chi2inv(0.95, 4);
P95_NIS  = chi2inv(0.95, 2);

% ----------------- main simulation loop ---------------------------------
x_true = [];

while true
    % --- simulate true motion -------------------------------------------
    x_true = getStateRect(x_true, T, V_CONST);   % provided by exercise

    % update state history
    X_Hist = addHistory(X_Hist, x_true);

    % --- measurement ----------------------------------------------------
    z = getMeasurement(x_true);                  % provided by exercise
    Z_Hist = addHistory(Z_Hist, z);

    % --- filter initialization ------------------------------------------
    if isempty(x_est)
        x_est = [z(1); z(2); 0; 0];  % initial guess
    end

    % ----------------- α-β tracker --------------------------------------
    % Prediction
    x_pred = F * x_est;

    % Covariance prediction
    P_pred = F * P * F' + Q;

    % Innovation (measurement residual)
    v = z - H * x_pred;          % 2x1

    % Innovation covariance (for NIS)
    S = H * P_pred * H' + R;

    % State update with constant α-β gain
    x_est = x_pred + K * v;      % 4x1

    % Covariance update with fixed gain K
    I = eye(4);
    P = (I - K * H) * P_pred * (I - K * H)' + K * R * K';

    % update estimation history
    X_est_Hist = addHistory(X_est_Hist, x_est);

    % ----------------- NEES & NIS ---------------------------------------
    % use only position part for NEES (x_true is [y; z])
    e_pos = x_true(1:2) - x_est(1:2);        % 2x1 position error

    k_step = k_step + 1;
    idx = mod(k_step - 1, HIST_SIZE) + 1;

    % NEES on position only: e_pos' * P_yy^{-1} * e_pos
    P_pos = P(1:2, 1:2);                     % 2x2 position covariance
    NEES_Hist(idx) = e_pos' * (P_pos \ e_pos);

    % NIS: v' S^{-1} v  (already 2D, OK)
    NIS_Hist(idx) = v' * (S \ v);

    % ----------------- Visualisation ------------------------------------
    % true and estimated trajectory
    subplot(2,2,1)
    plot(x_true(1), x_true(2), 'b.', 'MarkerSize', 25);
    hold on;
    plot(X_Hist(1,:), X_Hist(2,:), 'r-');
    plot(x_est(1), x_est(2), 'c.', 'MarkerSize', 25);
    plot(X_est_Hist(1,:), X_est_Hist(2,:), 'g-');
    hold off;
    daspect([1 1 1]);
    axis([-40, 40, 10, 80]);
    grid on;
    title('true & estimated trajectory');

    % measurement trajectory
    subplot(2,2,2);
    plot(z(1), z(2), 'r.', 'MarkerSize', 25);
    hold on;
    plot(Z_Hist(1,:), Z_Hist(2,:), 'r-', 'LineWidth', 1);
    daspect([1 1 1]);
    axis([-40, 40, 10, 80]);
    hold off;
    grid on;
    title('measurement');

    % NEES
    subplot(2,2,3)
    plot(NEES_Hist, 'LineWidth', 3);
    hold on;
    line([1 HIST_SIZE], [P95_NEES P95_NEES], 'Color', 'r');
    hold off;
    title('normalized estimation error squared (NEES)');
    xlabel('time steps (circular buffer)')

    % NIS
    subplot(2,2,4)
    plot(NIS_Hist, 'LineWidth', 3);
    hold on;
    line([1 HIST_SIZE], [P95_NIS P95_NIS], 'Color', 'r');
    hold off;
    title('normalized innovation squared (NIS)');
    xlabel('time steps (circular buffer)')

    drawnow;
    pause(0.05);
end


% ------------------------------------------------------------------------
function Hist = addHistory(Hist, val)
global HIST_SIZE;
if isempty(Hist)
    for i = 1:HIST_SIZE
        Hist(:, i) = val;
    end
end

Hist = circshift(Hist', 1)';
Hist(:,1) = val;