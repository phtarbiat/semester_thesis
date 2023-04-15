function [x_out,numIterations,errorFlag,normFun,numFunEva,numJacEva] = doglegRahpeymail(fun,jac,x_0,errorMargin,maxIteration)
% Implementation of a conjugated Trust-Region Method with the Dogleg-Method for
% solving the Trust-Region-Subproblem
%
% The implementation is based on Algorithm 2 in [9] with the Dogleg-Method
% from Procedure 11.6 in [8]
%
% Philipp Tarbiat
% Technical University of Munich
% 03/2022

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

op_k = zeros(3,1);

% Iteration loop
for nIterations = 1:maxIteration        
    % Update Trust-Region-Radius
    delta_k = normFun_k;
    
    %% Approximate solution of the Subproblem by the Dogleg-Method   
    % Calculate Cauchy-Point
    normG_k = norm(g_k,2);
    tau = min(1, (normG_k^3) / (delta_k * g_k' * B_k * g_k));
    pC_k = -tau * (delta_k / normG_k) * g_k;
    
    if abs(norm(pC_k,2) - delta_k) < eps
        p_k = pC_k;
    else
        % Calculate Newton-Step
        pJ_k = - jac_k \ fun_k;
        
        if any(~isfinite(pJ_k))
            p_k = pC_k;
        else
            if norm(pJ_k,2) <= delta_k
                p_k = pJ_k;
            else
                % If the Cauchy-Point is inside or outside the Trust-Region 
                % and the Newton-Step is not inside the Trust-Region, 
                % calculate a new step that is on the Trust-Region
                % Solving p_k = pc_k + tau*(pJ_k - pc_k) with norm(p_k) == delta
                % analytically for the l2-norm (p-q-Formel)
                p_diff = pJ_k - pC_k;
                
                p = (2 * pC_k' * p_diff) / (p_diff' * p_diff);
                q = ((pC_k' * pC_k) - delta_k^2) / (p_diff' * p_diff);

                a = -(0.5 * p);
                b = sqrt((0.25 * p^2) - q);
                
                if -a + b >= 0
                    tau = -a + b;
                else
                    tau = -a - b;
                end
                
                if ~isfinite(tau)
                    p_k = pC_k;
                else
                    p_k = pC_k + (tau * (pJ_k - pC_k));
                end
            end
            
            mpC = (g_k' * pC_k) + (0.5 * (pC_k' * B_k * pC_k));
            mp = (g_k' * p_k) + (0.5 * (p_k' * B_k * p_k));
            
            if mpC < mp
                p_k = pC_k;
            end
        end
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

    og_k = g_k;
    % Gradient and hessian for model function
    g_k = jac_k' * fun_k;
    B_k = jac_k' * jac_k;
    
    % Save old step
    op_k = p_k;
    
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

%In case there is no approximation of the state x that solves F(x) = 0
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