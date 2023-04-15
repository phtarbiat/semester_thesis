function [x_out,numIterations,errorFlag,normFun,numFunEva,numJacEva] = doglegBfgs(fun,jac,x_0,errorMargin,maxIteration)
% Implementation of a Trust-Region Method with the BFGS jacobian 
% approximation and the Dogleg-Method for solving the 
% Trust-Region-Subproblem
%
% The implementation is based on Algorithm 1 in [13] with the Dogleg-Method
% from Procedure 11.6 in [8]
%
% Philipp Tarbiat
% Technical University of Munich
% 03/2022

% System dimension
n = 3;

% Counter for function evaluations
numFunEva = 0;
numJacEva = 0;

% Parameters for the algorithm given in [13] p.329
rho = 0.0001;
c = 0.1;
%gamma = 0.7;
pf = 0;

% Set initial position
x_k = x_0;

% Evaluate function at x_k and calculate norm
fun_k = fun(x_k);
numFunEva = numFunEva + 1;
normFun_k = norm(fun_k,2);

% Check if error of the merit-function is low enough to exit iteration
if normFun_k <= errorMargin
    %Set function outputs and exit function
    x_out = x_k;
    normFun = normFun_k;
    numIterations = 0;
    errorFlag = false;
    return;
end

% Evaluate jacobian at x_0
jac_k = jac(x_k);
numJacEva = numJacEva + 1;

% If jacobian at initial point x_0 is invertiable than use the
% jacobian as the initialization of B_k else choose the identity matrix
if det(jac_k) ~= 0
    B_k = jac_k;
else
    B_k = eye(n);
end

% Gradient and Hessian for model function
g_k = jac_k' * fun_k;
mB_k = jac_k' * jac_k;

% Iteration loop
for nIterations = 1:maxIteration    
    % Calculate Trust-Region-Radius
    delta_k = (c^pf) * normFun_k;
       
    %% Approximate solution of the Subproblem by the Dogleg-Method
    % Calculate Cauchy-Point
    normG_k = norm(g_k,2);
    tau = min(1, (normG_k^3) / (delta_k * g_k' * mB_k * g_k));
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
            
            mpC = (g_k' * pC_k) + (0.5 * (pC_k' * mB_k * pC_k));
            mp = (g_k' * p_k) + (0.5 * (p_k' * mB_k * p_k));
            
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
    Pred_k = -((g_k' * p_k) + (0.5 * p_k' * mB_k * p_k));
    rho_k = Ared_k / Pred_k;
    
    % Check if rho_k is good enough
    if rho_k < rho
        pf = pf + 1;
    else
        % Calculate new state and corresponding function value
        ox_k = x_k;
        ofun_k = fun_k;
        x_k = x_k + p_k;
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
       
        y_k = fun_k - ofun_k;
       
        % Update jacobian approximation
        if y_k' * p_k > 0
            s_k = x_k - ox_k;
           
            B_k = B_k - ((B_k * (s_k * s_k') * B_k) / (s_k' * B_k * s_k)) + ((y_k * y_k') / (y_k' * s_k));
            
            % Gradient and hessian for model function
            g_k = B_k' * fun_k;
            mB_k = B_k' * B_k;
        else
            % Do not change mB_k
            g_k = B_k' * fun_k;
        end
       
        pf = 0;     
        
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