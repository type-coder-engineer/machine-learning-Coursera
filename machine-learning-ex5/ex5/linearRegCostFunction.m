function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear 
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the 
%   cost of using theta as the parameter for linear regression to fit the 
%   data points in X and y. Returns the cost in J and the gradient in grad

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost and gradient of regularized linear 
%               regression for a particular choice of theta.
%
%               You should set J to the cost and grad to the gradient.
%
h = X * theta; 
J = J + sum((h(:,1) - y(:,1)) .* (h(:,1) - y(:,1)));

n = size(grad,1);
J = J + lambda * sum(theta(2:n).*theta(2:n));
J = J/m/2;

for j = 1:n
    for i = 1:m
        grad(j, 1) = grad(j, 1) + (h(i) - y(i))*X(i, j);
    end
    if j ~= 1
        grad(j, 1) = grad(j, 1) + lambda * theta(j);
    end
end

grad = grad / m;








% =========================================================================

grad = grad(:);

end
