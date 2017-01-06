function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
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

end
