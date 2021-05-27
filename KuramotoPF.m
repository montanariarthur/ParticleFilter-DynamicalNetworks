%% Particle filtering example for a network of Kuramoto oscillators
% For details on application, results and conclusions, please refer to:
% Particle filtering of dynamical networks: Highlighting observability issues, by Arthur N. Montanari & Luis A. Aguirre (Jan 2019)
clear all; close all; clc;

%% Initial definitions
% Simulation parameters
dt = 0.1;              % integration step
tmax = 250;            % maximum time simulation
tspan = [0 dt tmax];
t = 0:dt:tmax;         % integration time vector
N = length(t);         % data points from "continuous" process

% Graph of the network
n = 15                                              % number of nodes (equal to the number of states)
Adj = diag(ones(n-1,1),-1) + diag(ones(n-1,1),+1);  % adjacency matrix (of a chain network)
rho = 0.1;                                          % coupling

% Observation: definition of sensor nodes
S = [1 n];              % set of sensor nodes, i.e. v_1 and v_n (the two ends of the chain)
q = length(S);          % number of outputs
                        % other sets of sensor nodes can be set as [v1 v2 ... vn]

% Definition of output matrix C, based on S. You don't need to change this.
C = zeros(q,n);
for i = 1:q
    C(i,S(i)) = 1;
end

% Natural frequencies of the oscillators    
omega = ones(n,1) + randn(n,1)*0.1;         % taken randomly around 1rad/s
omega = linspace(0.95,1.05,n)';             % taken linearly around 1rad/s
omega = [1.01552909768905;0.964845255242351;1.02175765727282;1.04759202408593;1.03145553401186;1.00232118227177;0.991525174499286;1.02151864750354;0.962676618094108;0.974474944611283;0.960118509058552;1.00209740946814;1.03754952446796;1.04965678624317;1.01596094244534];
                                            % saved from a random draw

% PF parameters
Np = 500;                   % number of particles
Nt = Np/2;                  % threshold
sigw = 0.1;                 % standard deviation of the process noise
sigv = 0.1;                 % standard deviation of the measurement noise
wm = omega + 0.0*rand(n,1); % frequency vector implemented in PF model
Nmc = 100;                  % number of Monte Carlo runs
Td = dt;                    % Euler's discretization time

% Inicialization
nrmse = zeros(n,Nmc);       % normalized RMSE
eta = zeros(n,1);           % performance index (per node/state)

%% Simulation of network of Kuramoto oscillators
% The following functions are used for numerical integration of
% continuous-time models.
x0 = 1*randn(1,n);      % initial conditions
[t,xt] = odeRK(@(t,x)kuramotonetwork(t,x,omega,Adj,rho),tspan,x0);

%% Particle filtering
tic
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
            x(k,:,p) = kuramotonetwork_discrete(x(k-1,:,p)',wm,Adj,rho,Td,sigw);
            
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
    rmse = sqrt(mean((xt-xhat).^2,1));       % RMSE of estimates
    deltaX = max(xt) - min(xt);              % data range
    nrmse(:,mc) = rmse./max(deltaX);              % Normalized RMSE
    
end  % end of the MC loop
eta = median(nrmse,2);                       % This is Eq. 17 in the paper.

%% Plot
figure(1)
subplot(211)
stem(1:n,omega,'black.');
xlim([0.5 15.5]); ylim([0.95 1.05])
ylabel('\omega_i')

subplot(212)
boxplot(nrmse',1:n,'Whisker',1);
hold on; plot(1:n,eta,'r');hold off; 
xlabel('v_i'); ylabel('\eta_i'); %ylim([0 0.07])

%% Functions

function dx = kuramotonetwork(t,s,w,Adj,rho)
% Implements the equations for a network of Kuramoto oscillators.
%    x = state variables of the full network of size n
%    w = column vector of parameters (natural frequency vector)
%    rho = coupling strength
%    Adj = Adjacency matrix. Adj(i,j)=1 indicates that there is a
%       directed link from node j to node i. This is a square matrix of
%       dimension length(x) = n
%    t = time vector
%    dx = time derivative of x (vector field at x)
n = length(s);
x(:,1) = s(1:n);
dx = w + sum( rho*sin(ones(n,1)*x' - x*ones(1,n)).*Adj , 2);
end

function xk = kuramotonetwork_discrete(s,w,Adj,rho,Td,sigw)
% Implements the discretized (via backward Euler) equations for a network
% of Kuramoto oscillators.
%    x = state variables of the full network of size n
%    w = column vector of parameters (natural frequency vector)
%    rho = coupling strength
%    Adj = Adjacency matrix. Adj(i,j)=1 indicates that there is a
%       directed link from node j to node i. This is a square matrix of
%       dimension length(x) = n
%    Td = discretization step
%    sigw = standard deviation of the process noise
%    xk = state iteration k (the following iteration)
n = length(s);
x(:,1) = s(1:n);
xk = x + Td*( w + sum ( rho*sin(ones(n,1)*x' - x*ones(1,n)).*Adj , 2) ) + randn(n,1)*sigw;
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

