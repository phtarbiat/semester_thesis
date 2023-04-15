function [x_out,numIterations,errorFlag,normFun,numFunEva,numJacEva] = broydenGood(fun,jac,x_0,errorMargin,maxIteration)
%Implementation of Broyden's Good Method
%
% The implementation is based on Algorithm 11.3 in [8] and the jacobian
% approximation update in [7]
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

% Evaluate function at x_0
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
% jacobian as the initialization of H_k else choose the identity matrix
if det(jac_0) ~= 0
    H_k = inv(jac_0);
else
    H_k = eye(n);
end

% Iteration loop
for nIterations = 1:maxIteration
    %% Find step direction   
    % Calculate state update
    p_k = -H_k * fun_k;

    % Save old state and function value
    x_kk = x_k;
    fun_kk = fun_k;
    
    %% Apply state update and update H_k
    x_k = x_kk + p_k;
    
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
    
    % Update approximation of the jacobian
    s_k = x_k - x_kk;
    y_k = fun_k - fun_kk;
    H_k = H_k + ((s_k - H_k*y_k) * (s_k' * H_k)) / (s_k' * H_k * y_k);
    
    % Break if H_k is not defined anymore
    if any(isnan(H_k),'all')
        % Set function outputs and exit function
        x_out = x_k;
        normFun = norm(fun_k);
        numIterations = nIterations;
        errorFlag = true;
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