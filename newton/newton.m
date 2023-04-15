function [x_out,numIterations,errorFlag,normFun,numFunEva,numJacEva] = newton(fun,jac,x_0,errorMargin,maxIteration)
% Implementation of the Newton-Raphson-Method
%
% The implementation is based on Algorithm 11.1 in [8]
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

% Check if error in the merit-function is low enough to exit iteration
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
    % Calculate state update
    p_k = -(jac_k \ fun_k);
    
    % If the jacobian is singular stop the algorithm
    if any(isinf(p_k))
        break;
    end
        
    %% Apply state update
    x_k = x_k + p_k;
    
    % evaluate function at x_k
    fun_k = fun(x_k);
    numFunEva = numFunEva + 1;

    % Check if error in the merit-function is low enough to exit iteration
    if norm(fun_k,2) <= errorMargin
        % set function outputs and exit function
        x_out = x_k;
        normFun = norm(fun_k,2);
        numIterations = nIterations;
        errorFlag = false;
        return;
    end

    % evaluate jacobian at x_k
    jac_k = jac(x_k);
    numJacEva = numJacEva + 1;
end

% in case there is no approximation of the state x that solves F(x) = 0
% found, give back the last state x_k and set the errorFlag to true
x_out = x_k;
normFun = norm(fun(x_k),2);
numIterations = maxIteration;
errorFlag = true;
end