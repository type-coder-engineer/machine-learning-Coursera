function g = sigmoid(z)
%SIGMOID Compute sigmoid function
%   g = SIGMOID(z) computes the sigmoid of z.

% You need to return the following variables correctly 
g = zeros(size(z));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the sigmoid of each value of z (z can be a matrix,
%               vector or scalar).
dimension = size(z);

if length(dimension) == 2
    for i = 1:dimension(1,1)
        for j = 1:dimension(1,2)
            g(i, j) = 1/(1 + exp(-z(i, j)));
        end
    end
else
    g = 1/(1 + exp(-z));
end


% =============================================================

end
