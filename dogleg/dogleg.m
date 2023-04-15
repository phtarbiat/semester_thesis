function [x_out,numIterations,errorFlag,normFun,numFunEva,numJacEva] = dogleg(fun,jac,x_0,errorMargin,maxIteration)
% Implementation of a Trust-Region Method with the Dogleg-Method for
% solving the Trust-Region-Subproblem
%
% The implementation is based on Algorithm 11.5 with the Dogleg-Method
% from Procedure 11.6 in [8]
%
% Philipp Tarbiat
% Technical University of Munich
% 03/2022

% Counter for function evaluations
numFunEva = 0;
numJacEva = 0;

% Parameters for Trust-Region
delta_k = 1;
delta_max = 1e10;
eta = 0.15;

% Set initial position
x_k = x_0;

% Evaluate function at x_k
fun_k = fun(x_k);
numFunEva = numFunEva + 1;

% Check if error of the merit-function is low enough to exit iteration
if norm(fun_k,2) <= errorMargin
    % Set function outputs and exit function
    x_out = x_k;
    normFun = norm(fun_k,2);
    numIterations = 0;
    errorFlag = false;
    return;
end

% Evaluate jacobi at x_k
jac_k = jac(x_k);
numJacEva = numJacEva + 1;

% Gradient and hessian for model function
g_k = jac_k' * fun_k;
B_k = jac_k' * jac_k;

% Iteration loop
for nIterations = 1:maxIteration        
    %% Approximate solution of the subproblem by the Dogleg-Method
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
    
    %% Verify if the found step is sufficient and update state and trust-region accordingly
    fun_pk = fun(x_k + p_k);
    numFunEva = numFunEva + 1;
    
    % Evaluate rho_k to see if the proposed step is sufficient
    Ared_k = 0.5 *((fun_k' * fun_k) - (fun_pk' * fun_pk));
    Pred_k = -((g_k' * p_k) + (0.5 * p_k' * B_k * p_k));
    rho_k = Ared_k / Pred_k;
    
    % Change Trust-Region-Radius according to rho_k
    if rho_k < 0.25
        delta_k = 0.25 * norm(p_k,2);
    else
        if rho_k > 0.75 && norm(p_k) == delta_k
            delta_k = min(2*delta_k , delta_max);
        end
    end
    
    delta_k = min(delta_k, delta_max);
    
    % Update x_k if rho_k is sufficient
    if rho_k > eta
        x_k = x_k + p_k;
        
        % Update function value
        fun_k = fun_pk;

        % Check if error of the merit-function is low enough to exit iteration
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
        
        %Gradient and hessian for model function
        g_k = jac_k' * fun_k;
        B_k = jac_k' * jac_k;
        
    end
    
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