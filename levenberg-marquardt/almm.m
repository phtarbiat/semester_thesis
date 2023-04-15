function [x_out,numIterations,errorFlag,normFun,numFunEva,numJacEva] = almm(fun,jac,x_0,errorMargin,maxIteration)
% Implementation of the Levenberg-Marquardt-Method with an adaptive choice
% for lambda
%
% The implementation is based on the algorithm proposed in [10] with an
% adaptive choice of lambda following Equation 1.6 in [15]
%
% Philipp Tarbiat
% Technical University of Munich
% 03/2022

% System dimension
n = 3;

% Counter for function evaluations
numFunEva = 0;
numJacEva = 0;

% Algorithm parameter
delta = 2;

% Set initial position
x_k = x_0;

% Evaluate function at x_k
fun_k = fun(x_k);
numFunEva = numFunEva + 1;

% Evaluate jacobian at x_k
jac_k = jac(x_k);
numJacEva = numJacEva + 1;

normG_k = norm(jac_k' * fun_k,2);

% Check if error of the merit-function is low enough to exit iteration
if norm(fun_k,2) <= errorMargin
    % Set function outputs and exit function
    x_out = x_k;
    normFun = norm(fun_k,2);
    numIterations = 0;
    errorFlag = false;
    return;
end

% Iteration loop
for nIterations = 1:maxIteration
    %% Find step direction    
    % Calculate lambda ([15] p.1243)
    if normG_k <= 1
        lambda_k = normG_k^delta;
    else
        lambda_k = normG_k^(-delta);
    end
    
    % Calculate state update
    d_k = -((jac_k' * jac_k + lambda_k * eye(n)) \ (jac_k' * fun_k));
    
    %% Apply state update
    x_k = x_k + d_k;
    
    % Evaluate function at x_k
    fun_k = fun(x_k);
    numFunEva = numFunEva + 1;
    
    % Evaluate jacobian at x_k
    jac_k = jac(x_k);
    numJacEva = numJacEva + 1;
    
    normG_k = norm(jac_k' * fun_k,2);
    
    % Check if error of the merit-function is low enough to exit iteration
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