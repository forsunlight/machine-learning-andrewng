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

size(Theta1);
size(Theta2);
m;
size(X);
size(y);
sum  = 0;




for i=1:m,
  a0_1 = 1;
  a1 = [1; (X(i, :))'];
  z2 = Theta1 * a1;
  a2 = [1; sigmoid(z2)];
  z3 = Theta2 * a2;
  a3 = sigmoid(z3);
  yi = zeros(num_labels, 1);
  yi(y(i)) = 1;
  
  for k = 1:num_labels
    a3_k = a3(k);
    yi_k = yi(k);
    temp = -1 * yi_k * log(a3_k) - (1 - yi_k) * log(1-a3_k);
    sum = sum + temp ;
  end;
  
end;



J = sum / m;



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% regulation
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
reg_theta1 = 0;
for j = 1: hidden_layer_size,
  for k = 2: input_layer_size+1,
    reg_theta1 = reg_theta1 + Theta1(j, k)^2;
  end;  
end;

reg_theta2 = 0;

for j = 1: num_labels,
  for k = 2: hidden_layer_size + 1,
    reg_theta2 = reg_theta2 + Theta2(j, k) ^2;
  end;  
end;

J = J + lambda * (reg_theta1 + reg_theta2) / (2 * m);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

for t = 1 : m
    a_1 = X(t, :); % [1 * 400]
    z_2 = [1, a_1] * Theta1'; % [1 * 401] * [401 * 25]
    a_2 = sigmoid(z_2); % [1 * 25]
    z_3 = [1, a_2] * Theta2'; % [1 * 26] * [26 * 10]
    a_3 = sigmoid(z_3); % [1 * 10]

    delta3 = a_3; % [1 * 10]
    delta3(y(t)) = delta3(y(t)) - 1;

    delta2 = delta3 * Theta2; % [1 * 10] * [10*26]
    delta2 = delta2(2 : end); % [1 * 25]
    delta2 = delta2 .* sigmoidGradient(z_2); % [1 * 25] .* [1 * 25]

    Theta2_grad = Theta2_grad + delta3' * [1, a_2]; % [26 * 1] * [1 * 10]
    Theta1_grad = Theta1_grad + delta2' * [1, a_1]; % [401 * 1] * [1 * 25]
end
Theta2_grad = Theta2_grad / m;
Theta1_grad = Theta1_grad / m;

Theta2_tmp = Theta2;
Theta1_tmp = Theta1;
Theta2_tmp(:, 1) = 0;
Theta1_tmp(:, 1) = 0;
Theta2_grad = Theta2_grad + lambda / m * Theta2_tmp;
Theta1_grad = Theta1_grad + lambda / m * Theta1_tmp;

% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
