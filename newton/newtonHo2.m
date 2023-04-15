function [x_out,numIterations,errorFlag,normFun,numFunEva,numJacEva] = newtonHo2(fun,jac,x_0,errorMargin,maxIteration)
% Implementation of 2nd order Newton-like line-search
%
% The implementation is based on Algorithm 2.2 in [12] with the general 
% line-search structure following [8]
%
% Philipp Tarbiat
% Technical University of Munich
% 03/2022

% Counter for function evaluations
numFunEva = 0;
numJacEva = 0;

% Set initial position
x_k = x_0;

% Evaluate function at x_k
fun_k = fun(x_k);
numFunEva = numFunEva + 1;

% Check if error in merit-function is low enough to exit iteration
if norm(fun_k,2) <= errorMargin
    % Set function outputs and exit function
    x_out = x_k;
    normFun = norm(fun_k,2);
    numIterations = 0;
    errorFlag = false;
    return;
end

% Evaluate jacobian at x_k
jac_k = jac(x_k);
numJacEva = numJacEva + 1;

% Iteration loop
for nIterations = 1:maxIteration
    %% Find step direction
    % Invert jacobi
    inverseJac_k = inv(jac_k);
    % If the jacobian is singular stop the algorithm
    if any(isinf(inverseJac_k))
        break;
    end  
    
    % Calculate first step and the corresponding state and function value
    d1_k = -(inverseJac_k * fun_k);
    y_k = x_k + d1_k;
    fun_yk = fun(y_k);
    numFunEva = numFunEva + 1;
    
    % Calculate second step
    d2_k = -(inverseJac_k * fun_yk);
    
    %% Apply state update
    x_k = y_k + d2_k;
    
    % Evaluate function at x_k
    fun_k = fun(x_k);
    numFunEva = numFunEva + 1;

    % Check if error in merit-function is low enough to exit iteration
    if norm(fun_k,2) <= errorMargin
        % Set function outputs and exit function
        x_out = x_k;
        normFun = norm(fun_k,2);
        numIterations = nIterations;
        errorFlag = false;
        return;
    end

    % Evaluatejacobian at x_k
    jac_k = jac(x_k);
    numJacEva = numJacEva + 1;
end

% In case there is no approximation of the state x that solves F(x) = 0
% found give back the last state x_k and set the error to true
x_out = x_k;
normFun = norm(fun(x_k),2);
numIterations = maxIteration;
errorFlag = true;
end