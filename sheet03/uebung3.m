function uebung3
close all;

% --- Konfiguration ---
global HIST_SIZE;
HIST_SIZE = 100;

% V_CONST = 0 setzt die Simulation auf beschleunigte Bewegung (Task 3) 
V_CONST = 0; 

% Zeit zwischen Messungen
T = 0.02;

% --- Kalman-Filter Variablen ---
x_pred = [];                    % Prädizierter Zustand (6x1)
x_est  = [];                    % Geschätzter Zustand x_est = (y,z,vy,vz,ay,az)
P_pred = [];                    % Prädizierte Kovarianz
P_est  = [];                    % Geschätzte Kovarianz (6x6)

% Historien-Buffer
X_Hist = [];                    
X_est_Hist = [];                
Z_Hist = [];                    
NEES_Hist = zeros(1,HIST_SIZE);     
NIS_Hist = zeros(1,HIST_SIZE);      

% --- 95% Consistency Chi-squared Bounds (Task 2) ---
% NEES hat nun 6 Freiheitsgrade (da Zustandvektor Dim 6 hat) 
P95_NEES_val = chi2inv(0.95, 6); 
% NIS hat weiterhin 2 Freiheitsgrade (da Messung Dim 2 hat)
P95_NIS_val = chi2inv(0.95, 2);

% --- Systemmatrizen für Constant Acceleration Model (Task 1) ---
% Zustandsreihenfolge: y, z, vy, vz, ay, az
% p_new = p + v*T + 0.5*a*T^2
% v_new = v + a*T
% a_new = a
F = [1 0 T 0 0.5*T^2 0; 
     0 1 0 T 0 0.5*T^2; 
     0 0 1 0 0.5*T^2 0; % Hinweis: In manchen Modellen nur T, hier 
     0 0 0 1 0 T;       % korrekte Integration v = v + a*T
     0 0 0 0 1 0;
     0 0 0 0 0 1];
% Korrektur Zeile 3 und 4 in F für Geschwindigkeit: v = v + a*T
F(3,5) = T; 
F(4,6) = T;

% Measurement matrix (H) - Wir messen nur y und z (Indizes 1 und 2)
H = [1 0 0 0 0 0; 
     0 1 0 0 0 0]; 

% Measurement noise covariance (R)
R = [1^2 0; 
     0 1^2]; 

% Process noise covariance (Q) - Task 1 & 3
% Modellierung als "Discrete White Jerk" (Ruck)
% Experimentell bestimmen: Bei V_CONST=0 (beschleunigt) muss q höher sein,
% um die Kurvenmanöver abzufangen.
q_val = 200; % Experimenteller Wert 

% Q Matrix Konstruktion für Constant Acceleration (Interleaved für y, z)
% Basis-Matrix für eine Dimension (y oder z):
% [T^5/20  T^4/8   T^3/6]
% [T^4/8   T^3/3   T^2/2]
% [T^3/6   T^2/2   T    ]
Q_dim = [T^5/20  T^4/8   T^3/6;
         T^4/8   T^3/3   T^2/2;
         T^3/6   T^2/2   T];

% Füllen der 6x6 Matrix
Q = zeros(6,6);
% Y-Dimension (Indizes 1, 3, 5)
Q([1,3,5], [1,3,5]) = Q_dim;
% Z-Dimension (Indizes 2, 4, 6)
Q([2,4,6], [2,4,6]) = Q_dim;

Q = Q * q_val; 

I_state = eye(size(F,1)); 

% Loop Variablen
k_cycle = 0;
x_true = [];

while (1)   
  k_cycle = k_cycle + 1;
  
  % --- Simulation ---
  % Hier wird getStateRect mit 6 Rückgabewerten verwendet (gemäß deiner Vorgabe)
  x_true = getStateRect(x_true, T, V_CONST);      
  
  X_Hist = addHistory(X_Hist, x_true);
  z = getMeasurement(x_true);
  Z_Hist = addHistory(Z_Hist, z);
  
  %=============== KALMAN FILTER START ==================================%
  
  NEES = 0;
  NIS = 0; 
  
  % Initialisierung
  if (isempty(x_est))   
    % Initial State: Position aus Messung, Rest 0
    % [y; z; vy; vz; ay; az]
    x_est = [z(1); z(2); 0; 0; 0; 0];
    
    % Initiale Kovarianz P (große Unsicherheit für v und a)
    P_est = diag([R(1,1), R(2,2), 100, 100, 10, 10]); 
    
    X_est_Hist = addHistory(X_est_Hist, x_est);
    continue;
  end
  
  % --- 1. Prediction Step ---
  x_pred = F * x_est;
  P_pred = F * P_est * F' + Q;
  
  % --- 2. Update Step ---
  y_res = z - H * x_pred; % Innovation (Residuum)
  
  % Innovation Covariance S
  S = H * P_pred * H' + R;
  
  % Kalman Gain K
  K = P_pred * H' / S;
  
  % Update State
  x_est = x_pred + K * y_res;
  
  % Update Covariance (Joseph Form für numerische Stabilität)
  I_KH = I_state - K * H;
  P_est = I_KH * P_pred * I_KH' + K * R * K';
  
  % --- Consistency Checks (Task 2) ---
  % NIS Calculation
  NIS = y_res' / S * y_res; 
  
  % NEES Calculation
  % Fehlervektor (nun 6 Dimensionen)
  e = x_true - x_est; 
  NEES = e' / P_est * e;
  
  %================ KALMAN FILTER END ===================================%
  
  % History Update
  X_est_Hist = addHistory(X_est_Hist, x_est);
  NEES_Hist = addHistory(NEES_Hist, NEES);
  NIS_Hist = addHistory(NIS_Hist, NIS);
  
  %=======================================================================%
  %     Visualisation
  %=======================================================================%
 
  % --- Trajectories ---
  subplot(2,2,1)
  plot(x_true(1),x_true(2),'b.','MarkerSize',20); hold on;
  plot(X_Hist(1,:),X_Hist(2,:),'r-'); 
  plot(x_est(1),x_est(2),'c.','MarkerSize',20); 
  plot(X_est_Hist(1,:),X_est_Hist(2,:),'g-'); 
  
  % Task 4: Visualisierung der Unsicherheit (Ellipse) [cite: 38]
  % Extrahiere 2x2 Kovarianz für Position (y,z)
  P_pos = P_est(1:2, 1:2);
  plotErrorEllipse(x_est(1:2), P_pos, 0.95); % Hilfsfunktion unten
  
  hold off;
  daspect([1 1 1]);
  axis([-40,40,10,80]);
  grid on
  title('True vs Estimated (Task 4: Ellipse included)')
  legend('True Pos', 'True Path', 'Est Pos', 'Est Path', 'Uncertainty');
  
  % --- Measurements ---
  subplot(2,2,2);
  plot(z(1),z(2),'r.','MarkerSize',15); hold on;  
  plot(Z_Hist(1,:),Z_Hist(2,:),'r-', 'LineWidth', 1); 
  daspect([1 1 1]);
  axis([-40,40,10,80]);
  hold off; grid on
  title('Measurements')
  
  % --- NEES ---  
  subplot(2,2,3)
  plot(NEES_Hist,'LineWidth',2); hold on;  
  line([0 size(NEES_Hist,2)], [P95_NEES_val P95_NEES_val], 'Color', 'r', 'LineStyle', '--');      
  hold off; grid on;
  title(['NEES (dof=6, 95% bound=' num2str(P95_NEES_val,'%.2f') ')']);
  
  % --- NIS ---
  subplot(2,2,4)
  plot(NIS_Hist,'LineWidth',2); hold on;
  line([0 size(NIS_Hist,2)], [P95_NIS_val P95_NIS_val], 'Color','r', 'LineStyle', '--');    
  hold off; grid on;
  title(['NIS (dof=2, 95% bound=' num2str(P95_NIS_val,'%.2f') ')']);
  
  drawnow
  pause(0.02);
end
end

% --- Helper Functions ---

function Hist = addHistory(Hist, val)
    global HIST_SIZE;
    if isempty(Hist)
      for i=1:HIST_SIZE
         Hist(:,i) = val;
      end
    end
    Hist = circshift(Hist',1)';       
    Hist(:,1) = val; 
end

% Task 4: Zusatzaufgabe - Unsicherheitsellipse [cite: 38], 
function plotErrorEllipse(mu, Sigma, p)
    % mu: Mittelwert [y; z]
    % Sigma: Kovarianzmatrix 2x2
    % p: Konfidenzintervall (z.B. 0.95)
    
    [eigVec, eigVal] = eig(Sigma);
    
    % Sortierung, damit Hauptachse korrekt ist
    [d, ind] = sort(diag(eigVal), 'descend');
    eigVal = eigVal(ind, ind);
    eigVec = eigVec(:, ind);
    
    % Skalierung basierend auf Chi-Quadrat Verteilung für 2 DOFs
    s = -2 * log(1 - p); 
    scale = sqrt(s);
    
    % Parameter t für die Ellipse
    t = linspace(0, 2*pi, 100);
    
    % Einheitskreis transformieren
    xy = [cos(t); sin(t)];
    
    % Skalieren mit Eigenwerten und Rotieren mit Eigenvektoren
    XY = eigVec * (sqrt(eigVal) * scale * xy);
    
    % Verschieben zum Mittelwert
    XY = XY + mu;
    
    plot(XY(1,:), XY(2,:), 'm-', 'LineWidth', 1.5);
end