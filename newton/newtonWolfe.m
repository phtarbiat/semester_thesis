function [x_out,numIterations,errorFlag,normFun,numFunEva,numJacEva] = newtonWolfe(fun,jac,x_0,errorMargin,maxIteration)
% Implementation of the Newton-Raphson-Method with adaptive step length 
% based on the weak Wolfe condition determined with inexact line-search
%
% The implementation is based on Algorithm 11.4 in [8] with Algorithm 11.1,
% Algorithm 3.5 and Algorithm 3.6 ([8])
%
% Philipp Tarbiat
% Technical University of Munich
% 03/2022

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
    % Calculate state update
    p_k = -(jac_k \ fun_k);
    
    % If the jacobian is singular stop the algorithm
    if any(isinf(p_k))
        break;
    end    

    %% Adapt step size based on weak Wolfe condition
    [alpha_k, error, numFunEva, numJacEva] = getWolfeStep(fun,jac,fun_k,jac_k,x_k,p_k,numFunEva,numJacEva);
    
    % If no sufficient step is found exit root finding
    if error
        break;
    end
        
    %% Apply state update
    x_k = x_k + alpha_k * p_k;
    
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

function [alpha, error, numFunEva, numJacEva] = getWolfeStep(fun,jac,fun_k,jac_k,x_k,p_k,numFunEva,numJacEva)
% Finding a step that satisfies the Wolfe condition
% based on Algorithm 3.5 in [8] (p.60)
% Initialize alpha_i based on [8] (p.59)

error = false;

% Parameters for adaptive step size given by [8] (p.62)
alpha_max = 100;
c1 = 10e-4;
c2 = 0.9;

alpha_i = 1;
oalpha_i = 0;

% Calculate updates of the function and jacobi with supposed step
fun_alpha_i = fun(x_k + alpha_i*p_k);
numFunEva = numFunEva + 1;
jac_alpha_i = jac(x_k + alpha_i*p_k);
numJacEva = numJacEva + 1;

% Calculate all needed values of phi(alpha)
phi_0 = norm(fun_k,2)^2;
phi_alpha_i = norm(fun_alpha_i,2)^2;
phid_0 = fun_k' * jac_k * p_k;
phid_alpha_i = fun_alpha_i' * jac_alpha_i * p_k;

ophi_alpha_i = phi_alpha_i;

% Check if Wolfe condition is satisfied
if (phi_alpha_i <= phi_0 + c1*alpha_i*phid_0) && (phid_alpha_i >= c2*phid_0)
    alpha = alpha_i;
else

    nIterationsAlpha = 1;

    % Iteraively find alpha_k that satisfies Wolfe condition
    while alpha_i <= alpha_max

        if (phi_alpha_i > phi_0 + c1*alpha_i*phid_0) || ...
                (phi_alpha_i >= ophi_alpha_i && nIterationsAlpha > 1)

            [alpha, error, numFunEva, numJacEva] = alphaZoom(fun,jac,x_k,p_k,c1,c2,oalpha_i,alpha_i,phi_0,phid_0,numFunEva,numJacEva);

            % Exit function if no sufficient alpha is found
            if error
                return;
            end

            break;
        end

        if abs(phid_alpha_i) <= -c2*phid_0
            alpha = alpha_i;
            break;
        end

        if phid_alpha_i >= 0

            [alpha, error, numFunEva, numJacEva] = alphaZoom(fun,jac,x_k,p_k,c1,c2,oalpha_i,alpha_i,phi_0,phid_0,numFunEva,numJacEva);

            % Exit function if no sufficient alpha is found
            if error
                return;
            end

            break;
        end

        % Update alpha and phi_alpha_i
        oalpha_i = alpha_i;
        alpha_i = alpha_i + 1;

        ophi_alpha_i = phi_alpha_i;

        nIterationsAlpha = nIterationsAlpha + 1;

        phi_alpha_i = norm(fun_alpha_i,2)^2;
        phid_alpha_i = fun_alpha_i' * jac_alpha_i * p_k;
    end

    if alpha_i > alpha_max
        alpha = alpha_max;
    end
end
end

function [alpha_k, error, numFunEva, numJacEva] = alphaZoom(fun,jac,x,p,c1,c2,alphaLo,alphaHi,phi_0,phid_0,numFunEva,numJacEva)
% Implementation based on Algorithm 3.6 in [8] (p.61)

error = false;

phi_alphaLo = norm(fun(x + alphaLo*p),2)^2;

while true
   
    alpha_j = (alphaLo + alphaHi) / 2;
    
    fun_alpha_j = fun(x + alpha_j*p);
    numFunEva = numFunEva + 1;
    
    phi_alpha_j = norm(fun_alpha_j,2)^2;
    
    
    if (phi_alpha_j > phi_0 + c1*alpha_j*phid_0) || (phi_alpha_j >= phi_alphaLo)
        alphaHi = alpha_j;
    else
        jac_alpha_j = jac(x + alpha_j*p);
        numJacEva = numJacEva + 1;
        
        phid_alpha_j = fun_alpha_j' * jac_alpha_j * p;
        
        if abs(phid_alpha_j) <= -c2*phid_0
            alpha_k = alpha_j;
            return;
        end
        
        if phid_alpha_j * (alphaHi - alphaLo) >= 0
            alphaHi = alphaLo;
        end
        
        alphaLo = alpha_j;
        phi_alphaLo = phi_alpha_j;
    end
    
    % Break out of function, if there is no sufficient alpha found
    if abs(alphaHi - alphaLo) <= eps
        alpha_k = 0;
        error = true;
        return;
    end
end
end