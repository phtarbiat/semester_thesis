function [x_out,numIterations,errorFlag,normFun,numFunEva,numJacEva] = newtonModified(fun,jac,x_0,errorMargin,maxIteration)
% Implementation of the modified Newton-Raphson-Method
%
% The implementation is based on Algorithms 11.1 in [8] with the step update
% calculated according to [4] (p.448)
%
% Philipp Tarbiat
% Technical University of Munich
% 03/2033

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
    % Calculate first intermediate step
    inverseJac_k = inv(jac_k);
    d_k = -(inverseJac_k * fun_k);
    
    % If the jacobian is singular stop the algorithm
    if any(isinf(d_k))
        break;
    end    
    
    % Calculate state and function value for the intermediate step
    y_k = x_k + d_k;
    fun_yk = fun(y_k);
    numFunEva = numFunEva + 1;
    
    % Calculate second intermediate step
    dhut_k = -(inverseJac_k * fun_yk);
            
    %% Apply state update
    x_k = x_k + (d_k + dhut_k);
    
    % Evaluate function at x_k
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
    
    % Evaluate jacobian at x_k
    jac_k = jac(x_k);
    numJacEva = numJacEva + 1;
end

% In case there is no approximation of the state x that solves F(x) = 0
% found, give back the last state x_k and set the errorFlag to true
x_out = x_k;
normFun = norm(fun(x_k),2);
numIterations = maxIteration;
errorFlag = true;
end