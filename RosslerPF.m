%% Particle filtering example for a network of Rossler oscillators
% For details on application, results and conclusions, please refer to:
% Particle filtering of dynamical networks: Highlighting observability issues, by Arthur N. Montanari & Luis A. Aguirre (Jan 2019)
clear all; close all; clc;

%% Initial definitions
% Simulation parameters
dt = 0.01;             % integration step
tmax = 300;            % maximum time simulation
tspan = [0 dt tmax];
t = 0:dt:tmax;         % integration time vector
N = length(t);         % data points from "continuous" process

% Graph of the network
m = 15                                              % number of nodes
n = 3*m;                                            % number of states (each node has a 3-dimensional system)
Adj = diag(ones(m-1,1),-1) + diag(ones(m-1,1),+1);  % adjacency matrix (of a chain network)
Lap = Adj - diag(sum(Adj,2));                       % negative Laplacian matrix
rho = [0 0.1 0];                                    % coupling factor for of x, y and z variables.
      % Nodes are coupled only by the y variable, as in Eq. (15).

% Observation: definition of sensor nodes
S = [1 m];              % set of sensor nodes, i.e. v_1 and v_n (the two ends of the chain)
qnet = length(S);       % number of nodes measured
                        % other sets of sensor nodes can be set as [v1 v2 ... vn]
Snode = [0 1 0];        % set of variables measured INDEPENDENTLY at each node
                        % e.g., Snode = [0 1 1] measures the y and z variable independently
qnode = sum(Snode);     % number of outputs at each node

% Definition of output matrix C, based on S. You don't need to change this.
C = zeros(3*qnet,n);
j = 0;
for i = 1:3:3*qnet
    j = j + 1;
    C(i,S(j)) = Snode(1);
    C(i+1,S(j)+m) = Snode(2);
    C(i+2,S(j)+2*m) = Snode(3);
end
C(~any(C,2),:) = [];            % removes zero rows
q = size(C,1);                  % number of outputs for the whole network (states measured)
% PS: In this algorithm, the state vector is NOT organized as [x1 y1 z1,
% x2, y2, z2, ..., xm, ym, zm]' as notated in Eq. (15) of the paper.
% Instead, it is organized as [x1, ..., xm, y1, ..., ym, z1, ..., zm]'.
% The output matrix C is defined in this algorithm accordingly.

% Parameters of the Rossler systems
a = linspace(0.398-0.01, 0.398+0.01, m)';   % taken linearly around 0.398
b = 2; c = 4;                               % equal for all nodes

% PF parameters
Np = 300;                   % number of particles
Nt = Np/2;                  % threshold
sigw = 0.01;                % standard deviation of the process noise
sigv = 0.1;                 % standard deviation of the measurement noise
am = a + 0.0*rand(m,1);     % parameter 'a' vector implemented in PF model
bm = b; cm = c;             % parameters 'b' and 'c' implemented in PF model
Nmc = 100;                  % number of Monte Carlo runs
Td = dt;                    % Euler's discretization time

% Inicialization
nrmse = zeros(m,Nmc);       % normalized RMSE
eta = zeros(m,1);           % performance index (per node/state)

%% Simulation of network of Rossler systems
% The following functions are used for numerical integration of
% continuous-time models.
x0 = 1*randn(1,n);      % initial cxonditions
[t,xt] = odeRK(@(t,x)rosslernetwork(t,x,a,b,c,Lap,rho,m),tspan,x0);

% Computes the measured data range $\Delta x_i$ in Eq. (17)
for j = 1:m
    xt_norm(:,j) = sqrt(xt(:,j).^2 + xt(:,j+m).^2 + xt(:,j+2*m).^2);
    deltaX(j) = max(xt_norm(:,j)) - min(xt_norm(:,j));
end

%% Particle filtering

% Monte Carlo loop
for mc = 1:Nmc
    display(['MC iteration = ', num2str(mc)])
    
    % Inicializations
    x = zeros(N,n,Np);          % x_k^{(p)}, i.e. p-th particle at time instant k of a (n x 1) vector
    xhat = zeros(N,n);          % estimated state vector (filter output)
    logW = zeros(N,Np);         % vector of weights in log scale
    Wbar = zeros(N,Np);         % vector of normalized weights NOT in log scale          
    Wsum = zeros(N,1);
    Neff = zeros(N,1);          % effective sample (of particles) size
    
    W(1,:) = log(repmat(1/Np,1,Np));          % initial weight = log(1/Np)
    Wsum(1) = sum(exp(W(1,:)));               % sum(W) = 1
    
    % Initicialization of all particles (discrete-time) with the
    % continuous-time data. No noise is used to improve convergence, but
    % the impact of uncertainty in the data can be investigated tuning the
    % noise level.
    x(1,:,:) = xt(1,:,:) + 0*randn(1,n,Np);   % particles initial PDF

    % Equation (12) in the paper: measured output with noise
    y = C*xt' + sigv*randn(q,N);
    
    % Particle Filtering (Algorithm #2)
    for k = 2:N

        % for each particle do:
        for p = 1:Np
            % Step 1 of Algorithm #2: Particle propagation (discrete time).
            % This is Eq. (13) in the paper.
            x(k,:,p) = rosslernetwork_discrete(x(k-1,:,p)',am,bm,cm,Lap,rho,m,Td,sigw);
            
            % Step 2 of Algorithm #@: Weight update (in log scale for
            % higher precision). This is Eq. 14 in the paper.
            logW(k,p) = logW(k-1,p) - log((2*pi)^(q/2)*sigv^q) - 1/2*norm((C*x(k,:,p)' - y(:,k))/sigv)^2; 
        end
        
        % Step 3 of Algorithm #2 is to go to Step 3 to 6 of Algorithm #1
               
        % Step 3 of Algorithm #1: Weight update
        logW(k,:) = logW(k,:) - max(logW(k,:));      % avoids numerical issues
        Wsum(k) = sum(exp(logW(k,:)));                  % This is the denominator of Eq. 4 in the paper
        Wbar(k,:) = exp(logW(k,:))./Wsum(k);         % This is Eq. 4 in the paper. The normalized weights are NOT in log scale.
        
        % Step 4 of Algorithm #1. This is Eq. 5 in the paper.Resampling
        Neff(k) = 1/(norm(exp(logW(k,:))/Wsum(k)))^2;  
        
        % Step 5 of Algorithm #1: Resampling 
        if Neff(k) < Nt
            Ind = SystematicResamp(Wbar(k,:),Np);
            for j = 1:1:n
                x(k,j,:) = x(k,j,Ind);
            end
            logW(k,:) = zeros(1,Np);         % note: zeros, not ones, 'cause it is on logarithmic scale
            Wsum(k) = Np;
        end    
        
    end    % end of the PF loop
    
    % Step 6 of Algorithm #1: Filter output. This is Eq. 6 in the paper.
    x = permute(x,[1 3 2]);
    for j = 1:n
        xhat(:,j) = sum((x(:,:,j).*Wbar)')';
    end
    
    % Statistic
    for j = 1:m
        errornorm(:,j) = sqrt( (xt(:,j)-xhat(:,j)).^2 + (xt(:,j+m)-xhat(:,j+m)).^2 + (xt(:,j+2*m)-xhat(:,j+2*m)).^2 );
    end
    rmse(mc,:) = sqrt(mean(errornorm.^2,1));   % RMSE of estimates PER NODE
    nrmse(:,mc) = rmse(mc,:)./max(deltaX);     % Normalized RMSE
    
end  % end of the MC loop
eta = median(nrmse,2);                  % This is Eq. 17 in the paper.

%% Plot
figure(1)
subplot(211)
stem(1:m,a,'black.');
xlim([0.5 15.5]); ylim([0.388 0.408])
ylabel('a_i')

subplot(212)
boxplot(nrmse',1:m,'Whisker',1);
hold on; plot(1:m,eta,'r');hold off; 
xlabel('v_i'); ylabel('\eta_i'); %ylim([0 0.07])

%% Functions

function ds = rosslernetwork(t,s,a,b,c,Lap,rho,m)
% Implements the equations for a network of Rossler systems.
%    s = state variables, sorted as [x(1:m) y(1:m) z(1:m)]', of the full network of size m
%    a = column vector of parameter 'a'
%    b, c = fixed parameters for all the nodes
%    rho = coupling strength vector
%    Lap = Laplacian matrix. This is a square matrix of dimension m.
%    m = number of nodes
%    t = time vector
%    ds = time derivative of s (vector field at s)
x(:,1) = s(1:m);
y(:,1) = s(m+1:2*m);
z(:,1) = s(2*m+1:end);

dx = - y - z + rho(1)*Lap*x;
dy = x + a.*y + rho(2)*Lap*y;
dz = b + z.*(x - c) + rho(3)*Lap*z;

ds = [dx; dy; dz];
end

function Xk = rosslernetwork_discrete(s,a,b,c,Lap,rho,m,Td,sigw)
% Implements the discretized (via backward Euler) equations for a network
% of Rossler systems.
%    x = state variables, sorted as [x(1:m) y(1:m) z(1:m)]', of the full network of size m
%    a = column vector of parameter 'a'
%    b, c = fixed parameters for all the nodes
%    rho = coupling strength vector
%    Lap = Laplacian matrix. This is a square matrix of dimension m.
%    Td = discretization step
%    sigw = standard deviation of the process noise
%    xk = state iteration k (the following iteration)
x(:,1) = s(1:m);
y(:,1) = s(m+1:2*m);
z(:,1) = s(2*m+1:end);

xk = x + Td*(- y - z + rho(1)*Lap*x) + randn(m,1)*sigw;
yk = y + Td*(x + a.*y + rho(2)*Lap*y) + randn(m,1)*sigw;
zk = z + Td*(b + z.*(x - c) + rho(3)*Lap*z) + randn(m,1)*sigw;

Xk = [xk; yk; zk];
end

function Ind=SystematicResamp(W,Nsamp)
% Systematic resampling, by Alexandre Mesquita (2016).
%     W = particle weights
%     Nsamp = number of particles
WS=size(W,2);
Ind=zeros(1,Nsamp);
U=rand()/Nsamp;         % acceptance probability
l=0; i=0; j=1; k=1; Ind0=1:WS;
while U<1
    if l>U              % accepts sample with prob. U
        U=U+1/Nsamp;    % new acceptance prob.
        Ind(k)=j-1;     % stores chosen sample
        k=k+1;
    else                % mix samples order
        i=round(j+rand()*(WS-j));
        l=l+W(i);
        Ind0([j i])=Ind0([i j]);
        W(:,[j i])=W(:,[i j]);
        j=j+1;
    end
end
Ind=Ind0(Ind);
end

function [t,X] = odeRK(h_ode,T0,X0)
% Implements the numerical integration algorithm Runge Kuta 4th order. The
% function works like 'ode45' from Matlab.
%       h_ode: ode function handle
%           ex.: use '@rossler' for a function named 'rossler'
%       T0: vector with intial and final integration time [t0 dt tf]
%           ex.: use [0 0.0001 100] for t=0 to t=100, with step 0.0001
%       X0: vector with intial values size(X0)=(1,n), n: system order
%           ex.: X0=[2 1 1] for a 3rd order system
% This code was was adapted (by Leandro Freitas) from the LAA, 31/12/16

% time vector
t0 = T0(1);     % initial time
h = T0(2);      % integration step
tf = T0(end);   % final time
t = t0:h:tf;    % time vector

% interesting variables
n = length(X0); % system order
N = length(t);  % # of points

% initial state
X = [X0; zeros(N-1,n)];

for k=2:length(t)
    x0 = X(k-1,:);
    
    % 1st call
    xd=feval(h_ode,t(k),x0)';
    savex0=x0;
    phi=xd;
    x0=savex0+0.5*h*xd;
    
    % 2nd call
    xd=feval(h_ode,t(k)+0.5*h,x0)';
    phi=phi+2*xd;
    x0=savex0+0.5*h*xd;
    
    % 3rd call
    xd=feval(h_ode,t(k)+0.5*h,x0)';
    phi=phi+2*xd;
    x0=savex0+h*xd;
    
    % 4th call
    xd=feval(h_ode,t(k)+h,x0)';
    X(k,:) = savex0+(phi+xd)*h/6;  
end
end

