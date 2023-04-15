function [x_out,numIterations,errorFlag,normFun,numFunEva,numJacEva] = trustRegionRahpeymail(fun,jac,x_0,errorMargin,maxIteration)
% Implementation of a conjugated Trust-Region Method with an approximation 
% of the exact solution of the Trust-Region-Subproblem
%
% The implementation is based on Algorithm 2 in [9] with the solution of
% the Subproblem from Algorithm 4.3 in [8]
%
% Philipp Tarbiat
% Technical University of Munich
% 03/2022

% System dimension
n = 3;

% Counter for function evaluations
numFunEva = 0;
numJacEva = 0;

% Parameters for Trust-Region-Radius
delta_max = 1e10;
u_1 = 0.1;
u_2 = 0.9;
c_1 = 0.25;
c_2 = 0.3;
beta_min = 0.1;
beta_max = 1000;

% Set initial position
x_k = x_0;

% Evaluate function at x_k
fun_k = fun(x_k);
numFunEva = numFunEva + 1;
normFun_k = norm(fun_k,2);

% Check if error of the merit-function is low enough to exit iteration
if normFun_k <= errorMargin
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

og_k = zeros(3,1);

% Gradient and hessian for model function
g_k = jac_k' * fun_k;
B_k = jac_k' * jac_k;

% Calculate Gauss-Newton-Step
p_k = -B_k \ g_k;
normP_k = norm(p_k,2);

op_k = zeros(3,1);

% Iteration loop
for nIterations = 1:maxIteration        
    % Update Trust-Region-Radius
    delta_k = normFun_k;
    
    %% Approximate solution of the Subproblem by trying to find exact solution of the TR-subproblem and terminate after 3 steps
    if normP_k <= delta_k
        % p_k is Gauss-Newton-Step
    else   
        lambda_l = 1;

        % Stop after 3 Iterations because the found lambda should be
        % sufficient
        for l = 1:3
            [R, error] = chol(B_k);

            if error
                % Set function outputs and exit function
                x_out = x_0;
                normFun = norm(fun_k,2);
                numIterations = NaN;
                errorFlag = true;
                return;
            end

            p_l = -(R' * R) \ g_k;
            q_l = R' \ p_l;

            lambda_l = lambda_l + ((norm(p_l,2) / norm(q_l,2))^2 * ((norm(p_l,2) - delta_k) / delta_k));

        end

        p_k = -(B_k + lambda_l * eye(n)) \ g_k;
    
    end
    
    
    %% Verify if the found step is sufficient and update state and Trust-Region accordingly
    fun_pk = fun(x_k + p_k);
    numFunEva = numFunEva + 1;
    
    % Evaluate rho_k to see if the proposed step is sufficient
    Ared_k = 0.5 *((fun_k' * fun_k) - (fun_pk' * fun_pk));
    Pred_k = -((g_k' * p_k) + (0.5 * p_k' * B_k * p_k));
    rho_k = Ared_k / Pred_k;
    
    % Update x_k if rho_k is sufficient otherwise use cg and backtracking to find step
    if rho_k > u_1
        x_k = x_k + p_k;
        
        if rho_k >= u_2
           delta_k = c_2 * delta_k; 
           delta_k = min(delta_k, delta_max);
        end
    else
        delta_k = c_1 * delta_k;
        delta_k = min(delta_k, delta_max);
        
        y_k = g_k - og_k;
        
        beta_k = (g_k' * y_k) / (op_k' * y_k);
        beta_k = min(max(beta_k,beta_min),beta_max);
        
        if nIterations == 1
            p_k = -g_k;
        else
            p_k = -g_k + beta_k * p_k;
        end
        
        [alpha_k,error,numFunEva] = backtracking(fun,x_k,fun_k,p_k,g_k,numFunEva);

        % If no sufficient step is found exit root finding
        if error
            break;
        end
        
        x_k = x_k + alpha_k * p_k;
    end    
    
    % Update function value
    fun_k = fun_pk;
    normFun_k = norm(fun_k,2);

    % Check if error of the merit-function is low enough to exit iteration
    if normFun_k <= errorMargin
        %Set function outputs and exit function
        x_out = x_k;
        normFun = norm(fun_k,2);
        numIterations = nIterations;
        errorFlag = false;
        return;
    end

    % Evaluate jacobian at x_k
    jac_k = jac(x_k);
    numJacEva = numJacEva + 1;

    og_k = g_k;

    % Gradient and hessian for model function
    g_k = jac_k' * fun_k;
    B_k = jac_k' * jac_k;
    
    % Save old step
    op_k = p_k;

    % Calculate Gauss-Newton-Step
    p_k = -B_k \ g_k;
    normP_k = norm(p_k,2);
    
    % Check if Trust-Region-Radius becomes too small
    if delta_k <= errorMargin
        % Set function outputs and exit function
        x_out = x_k;
        normFun = norm(fun_k,2);
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

function [alpha,error,numFunEva] = backtracking(fun,x,fun_k,p,g,numFunEva)

gamma = 0.5;
sigma = 0.5;
error = false;

alpha_i = 1;

fun_alpha_i = fun(x + alpha_i * p);
numFunEva = numFunEva + 1;

while fun_alpha_i > fun_k + gamma * alpha_i * g' * p
    % Update alpha
    alpha_i = sigma * alpha_i;
    fun_alpha_i = fun(x + alpha_i * p);
    numFunEva = numFunEva + 1;

    % Break out of function, if there is no sufficient alpha found
    if alpha_i <= eps
        alpha = 0;
        error = true;
        return;
    end
end

alpha = alpha_i;
end