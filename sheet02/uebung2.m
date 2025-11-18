function uebung2
close all;
%history size: the number of measurements, estimations,... to store for
%visualization
global HIST_SIZE;
HIST_SIZE = 100;

%constant velocity, no acceleration if set to a value > 0
V_CONST = 0;

%the time betweeen to measurements we get
T = 0.02;

% Kalman-Filter variables
x_pred = [];                    % predicted state          
x_est  = [];                    % state estimation x_est = (y,z,vy,vz)
P_pred = [];                    % predicted error covariance
P_est  = [];                    % estimated error covariance

X_Hist = [];                    % state history
X_est_Hist = [];                % estimation history
Z_Hist = [];                    % measurement history

% performance variables
NEES_Hist = zeros(1,HIST_SIZE);     % NEES - history
NIS_Hist = zeros(1,HIST_SIZE);      % NIS - history

% --- 95% Consistency Chi-squared Bounds (Task 2) ---
P95_NEES_val = chi2inv(0.95, 4);
P95_NIS_val = chi2inv(0.95, 2);

F = [1 0 T 0; 
     0 1 0 T; 
     0 0 1 0; 
     0 0 0 1];

% Measurement matrix (H)
H = [1 0 0 0; 
     0 1 0 0]; 

% Measurement noise covariance (R)
R = [1^2 0; 
     0 1^2]; 

% Process noise covariance (Q)
q_val = 50;
Q_base = [0.25*T^4 0 0.5*T^3 0; 
          0 0.25*T^4 0 0.5*T^3; 
          0.5*T^3 0 T^2 0; 
          0 0.5*T^3 0 T^2];
Q = q_val * Q_base; 

I_state = eye(size(F,1)); 

% History Buffers
k_cycle = 0;
K_Hist_T3 = zeros(4, 2, 500);
P_diag_Hist_T3 = zeros(4, 500);

x_true = [];
while (1)   
  k_cycle = k_cycle + 1;
  
  % simulation loop  
  x_true = getStateRect(x_true,T, V_CONST);      

  % update state history
  X_Hist = addHistory(X_Hist, x_true);


  z = getMeasurement(x_true);
  % update measurement history  
  Z_Hist = addHistory(Z_Hist, z);

  %=============== insert your kalman filter ======================================%
  
  NEES = 0;
  NIS = 0; 

  % filter initialization (once per simulation)  
  if (isempty(x_est))   
    x_est = [z(1); z(2); 0; 0];
    P_est = diag([R(1,1), R(2,2), 10000, 10000]); 
    
    if k_cycle <= 500
        P_diag_Hist_T3(:, k_cycle) = diag(P_est);
    end
    X_est_Hist = addHistory(X_est_Hist, x_est);
    
    continue;
  end

  % --- 1. Prediction Step ---
  x_pred = F * x_est;
  P_pred = F * P_est * F' + Q;

  % --- 2. Update Step ---
  y = z - H * x_pred;
  
  % Calculate innovation covariance
  S = H * P_pred * H' + R;

  % Calculate NIS
  NIS = y' / S * y; % y' * inv(S) * y
  
  % Calculate Kalman Gain (K)
  K = P_pred * H' / S;
  
  % Update the state estimation (get x_est(k|k))
  x_est = x_pred + K * y;
  I_KH = I_state - K * H;
  P_est = I_KH * P_pred * I_KH' + K * R * K';

  if k_cycle <= 500
      K_Hist_T3(:, :, k_cycle) = K;
      P_diag_Hist_T3(:, k_cycle) = diag(P_est);
  end

  % Calculate NEES
  e = x_true(1:4) - x_est; 
  NEES = e' / P_est * e;
  %================================================================================%

  % update estimation history
  X_est_Hist = addHistory(X_est_Hist, x_est);

  % Add consistency check (Task 2)
  % Update the history buffers
  NEES_Hist = addHistory(NEES_Hist, NEES);
  NIS_Hist = addHistory(NIS_Hist, NIS);

  P95_NEES = P95_NEES_val;
  P95_NIS = P95_NIS_val;
  
  %=======================================================================%
  %     Visualisation
  %=======================================================================%
 
  % true and estimated trajectory
  subplot(2,2,1)
  plot(x_true(1),x_true(2),'b.','MarkerSize',25);  
  hold on;
  plot(X_Hist(1,:),X_Hist(2,:),'r-'); 
  plot(x_est(1),x_est(2),'c.','MarkerSize',25); 
  plot(X_est_Hist(1,:),X_est_Hist(2,:),'g-'); 
  hold off;
  daspect([1 1 1]);
  axis([-40,40,10,80]);
  grid on
  title 'true trajectory'

  % measurement trajectory
  subplot(2,2,2);
  plot(z(1),z(2),'r.','MarkerSize',25);  
  hold on;  
  plot(Z_Hist(1,:),Z_Hist(2,:),'r-', 'LineWidth', 1); 
  daspect([1 1 1]);
  axis([-40,40,10,80]);
  hold off;
  grid on
  title 'measurement'
  
  % NEES  
  subplot(2,2,3)
  plot(NEES_Hist,'LineWidth',3);
  hold on;  
  line([0 size(NEES_Hist,2)], [P95_NEES P95_NEES], 'Color', 'r');      
  hold off;
  title 'normalized estimation error squared (NEES)'

  % NIS
  subplot(2,2,4)
  plot(NIS_Hist,'LineWidth',3);
  hold on;
  title 'normalized innovation squared (NIS)'
  line([0 size(NIS_Hist,2)], [P95_NIS P95_NIS], 'Color','r');    
  hold off;  
  
  % --- Task 3: Plotting ---
  if k_cycle == 500
      t = 1:500;
      
      % Plot Kalman Gains
      figure('Name', 'Task 3: Kalman Gain (K) Components');
      subplot(2,2,1); plot(t, squeeze(K_Hist_T3(1,1,t))); 
      title('K(1,1) (y-meas -> y-pos)'); grid on;
      subplot(2,2,2); plot(t, squeeze(K_Hist_T3(3,1,t)));
      title('K(3,1) (y-meas -> vy-vel)'); grid on;
      subplot(2,2,3); plot(t, squeeze(K_Hist_T3(2,2,t)));
      title('K(2,2) (z-meas -> z-pos)'); grid on;
      subplot(2,2,4); plot(t, squeeze(K_Hist_T3(4,2,t)));
      title('K(4,2) (z-meas -> vz-vel)'); grid on;
      
      % Plot Variances
      figure('Name', 'Task 3: State Variances (Diagonal of P_est)');
      subplot(2,2,1); plot(t, P_diag_Hist_T3(1,:)); 
      title('Variance_y'); grid on; legend('Var(y)');
      subplot(2,2,2); plot(t, P_diag_Hist_T3(2,:));
      title('Variance_z'); grid on; legend('Var(z)');
      subplot(2,2,3); plot(t, P_diag_Hist_T3(3,:));
      title('Variance_vy'); grid on; legend('Var(vy)');
      subplot(2,2,4); plot(t, P_diag_Hist_T3(4,:));
      title('Variance_vz'); grid on; legend('Var(vz)');
  end
  
  drawnow
  pause(0.05);
end


function Hist = addHistory(Hist, val)
global HIST_SIZE;
if isempty(Hist)
  for i=1:HIST_SIZE
     Hist(:,i) = val;
  end
end

Hist = circshift(Hist',1)';       
Hist(:,1) = val;