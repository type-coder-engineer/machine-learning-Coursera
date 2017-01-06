function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
%size(Theta1)
%size(Theta2)
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%
X = [ones(m, 1), X];
h1 = sigmoid(X * Theta1');  % don't forget to implement the sigmoid function!!
h1 = [ones(m, 1), h1];
h2 = sigmoid(h1 * Theta2');

new_y = zeros(m, num_labels);
for k = 1:num_labels
    new_y(find(y == k), k) = 1;
end

for i = 1:m
    J = J + sum(-new_y(i, :).*log(h2(i, :)) - (1 - new_y(i, :)).*log(1 - h2(i, :))); 
end
J = J/m;

reg_part = 0;
for i = 1:size(Theta1,1)
    for j = 2:size(Theta1,2)
        reg_part = reg_part + Theta1(i, j)*Theta1(i, j);
    end
end

for i = 1:size(Theta2,1)
    for j = 2:size(Theta2,2)
        reg_part = reg_part + Theta2(i, j)*Theta2(i, j);
    end
end

reg_part = reg_part * lambda / 2 / m;
J = J + reg_part;

% cost part over

delta1 = zeros(size(Theta1, 1), size(Theta1, 2)); % 25*401
delta2 = zeros(size(Theta2, 1), size(Theta2, 2)); % 10*26
error3 = zeros(size(Theta2, 1), 1);
z2 = X * Theta1'; % 5000*26
a1 = X; % 5000*401
a2 = h1; % 5000*26

for i = 1:m
    for k = 1:size(Theta2, 1)
        error3(k, 1) = h2(i, k) - new_y(i, k);
    end
    g_z2 = [1, sigmoidGradient(z2(i, :))];
    %size(g_z2(:))
    %size(Theta2' * error3)
    error2 = Theta2' * error3 .* g_z2(:);
    %size(error2)
    %pause;
    delta1 = delta1 + error2(2:end) * a1(i, :); % be careful with the dimension
    % here we make (25x1)x(1x401) = (25x401), which equals to the dimension
    % of the delta1, the same reason afterwards
    delta2 = delta2 + error3 * a2(i, :);
    % don't forget that the gradient is for the every theta, including the
    % bias
end

Theta1_grad = delta1/m; % 25,401
for i = 1:size(Theta1, 1)
    for j = 2:size(Theta1, 2)
        Theta1_grad(i, j) = Theta1_grad(i, j) + lambda / m * Theta1(i, j);
    end
end
Theta2_grad = delta2/m; % 10, 26
for i = 1:size(Theta2, 1)
    for j = 2:size(Theta2, 2)
        Theta2_grad(i, j) = Theta2_grad(i, j) + lambda / m * Theta2(i, j);
    end
end

% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
