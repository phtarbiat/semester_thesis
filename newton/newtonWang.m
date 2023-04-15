function [x_out,numIterations,errorFlag,normFun,numFunEva,numJacEva] = newtonWang(fun,jac,x_0,errorMargin,maxIteration)
% Implementation of a three-step Newton-like line-search
%
% The implementation is based on Equation 6 in [11] with the general 
% line-search structure following [8]
%
% Philipp Tarbiat
% Technical University of Munich
% 03/2022

% System dimension
n = 3;

% Counter for function evaluations
numFunEva = 0;
numJacEva = 0;

% Set initial position
x_k = x_0;

% Evaluate function at x_k
fun_xk = fun(x_k);
numFunEva = numFunEva + 1;

% Check if Error in the merit-function is low enough to exit iteration
if norm(fun_xk,2) <= errorMargin
    % Set function outputs and exit function
    x_out = x_k;
    normFun = norm(fun_xk,2);
    numIterations = 0;
    errorFlag = false;
    return;
end

% Evaluate jacobian at x_k
invJac_xk = inv(jac(x_k));
numJacEva = numJacEva + 1;

% Iteration loop
for nIterations = 1:maxIteration
    %% Find step
    % Calculate first step and according function and jacobian value
    y_k = x_k - (invJac_xk * fun_xk);
    fun_yk = fun(y_k);
    numFunEva = numFunEva + 1;
    jac_yk = jac(y_k);
    numJacEva = numJacEva + 1;
    
    a = (2*eye(n) - invJac_xk * jac_yk);
    
    % Calculate second step and according function value
    z_k = y_k - (a * (invJac_xk * fun_yk));
    fun_zk = fun(z_k);
    numFunEva = numFunEva + 1;
    
    % Calculate thrid step
    x_k = z_k - (a * (invJac_xk * fun_zk));
    
    %% Check if approximation is good enough
    % Evaluate function at x_k
    fun_xk = fun(x_k);
    numFunEva = numFunEva + 1;

    % Check if error in the merit-function is low enough to exit iteration
    if norm(fun_xk,2) <= errorMargin
        % Set function outputs and exit function
        x_out = x_k;
        normFun = norm(fun_xk,2);
        numIterations = nIterations;
        errorFlag = false;
        return;
    end

    % Evaluate jacobian at x_k
    invJac_xk = inv(jac(x_k));
    numJacEva = numJacEva + 1;
end

% In case there is no approximation of the state x that solves F(x) = 0
% found, give back the last state x_k and set the errorFlag to true
x_out = x_k;
normFun = norm(fun(x_k),2);
numIterations = maxIteration;
errorFlag = true;
end