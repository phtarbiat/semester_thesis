function [x_out,numIterations,errorFlag,normFun,numFunEva,numJacEva] = mlm(fun,jac,x_0,errorMargin,maxIteration)
% Implementation of the Levenberg-Marquardt-Method with an adaptive choice
% for lambda
%
% The implementation is based on Algorithm 2.1 proposed in [4]
%
% Philipp Tarbiat
% Technical University of Munich
% 03/2022

% System dimension
n = 3;

% Counter for function evaluations
numFunEva = 0;
numJacEva = 0;

% Algorithm parameters (choosen after [4] p.463)
delta = 1;
p_0 = 0.0001;
p_1 = 0.25;
p_2 = 0.75;
u_k = 1e-5;
m = 1e-8;

% Set initial position
x_k = x_0;

% Evaluate function at x_k
fun_k = fun(x_k);
numFunEva = numFunEva + 1;

% Evaluate jacobian at x_k
jac_k = jac(x_k);
numJacEva = numJacEva + 1;

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
    % Calculate lambda
    lambda_k = u_k * norm(fun_k,2)^delta;
    
    jac_k2 = jac_k' * jac_k;
    
    % Calculate first intermediate step and corresponding state and function
    % value
    inverseTerm = inv(jac_k2 + lambda_k * eye(n));
    d_k = -(inverseTerm * (jac_k' * fun_k));
    y_k = x_k + d_k;
    
    fun_yk = fun(y_k);
    numFunEva = numFunEva + 1;
    
    % Calculate second intermediate step
    dhut_k = -(inverseTerm * (jac_k' * fun_yk));
    
    % Calculate full step
    s_k = d_k + dhut_k;
    
    fun_sk = fun(x_k + s_k);
    numFunEva = numFunEva + 1;
    
    %% Check if step is sufficient and update state and parameters accordingly
    % Calculate ratio of actual reduction
    Ared_k = norm(fun_k,2)^2 - norm(fun_sk,2)^2;
    Pred_k = norm(fun_k,2)^2 - norm(fun_k + jac_k * d_k,2)^2 + ...
             norm(fun_yk,2)^2 - norm(fun_yk + jac_k * dhut_k,2)^2;
    r_k = Ared_k / Pred_k;
    
    % Decide if reduction is suffiecient and change state accordingly
    if r_k >= p_0
        x_k = x_k + s_k;
        
        fun_k = fun_sk;
        jac_k = jac(x_k);
        numJacEva = numJacEva + 1;

        % Check if error of the merit-function is low enough to exit iteration
        if norm(fun_k,2) <= errorMargin
            % Set function outputs and exit function
            x_out = x_k;
            normFun = norm(fun_k,2);
            numIterations = nIterations;
            errorFlag = false;
            return;
        end
    else
        % x_k does not change
    end
    
    % Change u_k according to ratio of actual reduction
    if r_k < p_1
        u_k = 4 * u_k;
    elseif r_k > p_2
        u_k = max(0.25 * u_k, m);
    else
        % u_k does not change
    end    
end

% In case there is no approximation of the state x that solves F(x) = 0
% found give back the last state x_k and set the error to true
x_out = x_k;
normFun = norm(fun(x_k),2);
numIterations = maxIteration;
errorFlag = true;
end