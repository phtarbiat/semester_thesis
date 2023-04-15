function [x_out,numIterations,errorFlag,normFun,numFunEva,numJacEva] = newtonStationary(fun,jac,x_0,errorMargin,maxIteration)
%Implementation of the Newton-Stationary.Method
%
% The implementation is based on Algorithm 11.3 in [8] and the jacobian
% approximation update in [7]
%
% Philipp Tarbiat
% Technical University of Munich
% 03/2022

% Counter for function evaluations
numFunEva = 0;
numJacEva = 0;

% Set initial position
x_k = x_0;

%E valuate function at x_0
fun_k = fun(x_0);
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

% Evaluate jacobian at x_0
jac_0 = jac(x_0);
numJacEva = numJacEva + 1;

% If jacobian at initial point x_0 is invertiable then use the inverse
% jacobian as the initialization of H_k else quit
if det(jac_0) ~= 0
    H_k = inv(jac_0);
else
    x_out = x_k;
    normFun = norm(fun(x_k),2);
    numIterations = NaN;
    errorFlag = true;
    return;
end

% Iteration loop
for nIterations = 1:maxIteration
    %% Find step direction   
    % Calculate state update
    p_k = -H_k * fun_k;
    
    %% Apply state update and update H_k
    x_k = x_k + p_k;
    
    % Calculate new function value
    fun_k = fun(x_k);
    numFunEva = numFunEva + 1;
    
    % Check if error in the merit-function is low enough to exit iteration
    if norm(fun_k,2) <= errorMargin
        % Set function outputs and exit function
        x_out = x_k;
        normFun = norm(fun_k,2);
        numIterations = nIterations;
        errorFlag = false;
        return;
    end
end

% In case there is no approximation of the state x that solves F(x) = 0
% found give back the last state x_k and set the error to true
x_out = x_k;
normFun = norm(fun(x_k),2);
numIterations = maxIteration;
errorFlag = true;
end