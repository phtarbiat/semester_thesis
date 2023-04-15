function [x_out,numIterations,errorFlag,normFun,numFunEva,numJacEva] = trustRegionFan(fun,jac,x_0,errorMargin,maxIteration)
% Implementation of a Trust-Region Method with an adaptive 
% Trust-Region-Radius and an approximation of the exact solution of the 
% Trust-Region-Subproblem
%
% The implementation is based on Algorithm 2.1 in [6] with the solution of
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

% Parameters for adaptive Trust-Region-Radius from [6] p.69
c0 = 0.0001;
c2 = 0.25;
c4 = 0.75;
c5 = 0.25;
c6 = 8;
delta = 1;
M = 8;
u_k = 0.1;

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

% Evaluate jacobian at x_k
jac_k = jac(x_k);
numJacEva = numJacEva + 1;

% Gradient and hessian for model function
g_k = jac_k' * fun_k;
B_k = jac_k' * jac_k;

% Calculate Gauss-Newton-Step
p_k = -B_k \ g_k;
normP_k = norm(p_k,2);

% Iteration loop
for nIterations = 1:maxIteration
    
    % Calculate Trust-Region-Radius
    delta_k = u_k * (norm(fun_k,2)^delta);
       
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
    
    % Update x_k if rho_k is sufficient
    if rho_k > c0
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
        
        % Gradient and hessian for model function
        g_k = jac_k' * fun_k;
        B_k = jac_k' * jac_k;
        
        % Calculate Gauss-Newton-Step
        p_k = -B_k \ g_k;
        normP_k = norm(p_k,2);
        
    else
        % Do not change x_k
    end
    
    % Update u_k according to rho_k
    if rho_k < c2
        u_k = c5 * u_k;
    elseif rho_k >= c2 && rho_k <= c4
        % Do not change u_k
    else
        u_k = min(c6 * u_k, M);
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