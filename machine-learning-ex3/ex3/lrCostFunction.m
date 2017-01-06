function [J, grad] = lrCostFunction(theta, X, y, lambda)
%LRCOSTFUNCTION Compute cost and gradient for logistic regression with 
%regularization
%   J = LRCOSTFUNCTION(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta
%
% Hint: The computation of the cost function and gradients can be
%       efficiently vectorized. For example, consider the computation
%
%           sigmoid(X * theta)
%
%       Each row of the resulting matrix will contain the value of the
%       prediction for that example. You can make use of this to vectorize
%       the cost function and gradient computations. 
%
% Hint: When computing the gradient of the regularized cost function, 
%       there're many possible vectorized solutions, but one solution
%       looks like:
%           grad = (unregularized gradient for logistic regression)
%           temp = theta; 
%           temp(1) = 0;   % because we don't add anything for j = 0  
%           grad = grad + YOUR_CODE_HERE (using the temp variable)
%

h = sigmoid(X * theta);
dimension = size(theta);

for i = 1:m
    J = J + (-y(i, 1) * log(h(i, 1)) - (1 - y(i, 1)) * log(1 - h(i, 1)));
end
for j = 2:dimension(1,1)
    J = J + lambda / 2 * theta(j, 1)*theta(j, 1);
end
J = J / m;

for i = 1:dimension(1,1)
    for j = 1:dimension(1,2)
        
        for k = 1:m
            grad(i, j) = grad(i, j) + (h(k, 1) - y(k, 1)) * X(k, i);
        end
        if i >= 2
            grad(i, j) = grad(i, j) + lambda * theta(i, 1);
        end
        grad(i, j) = grad(i, j) / m;
        
    end
end

% =============================================================

grad = grad(:);

end
