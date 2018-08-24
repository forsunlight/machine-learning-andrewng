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


h_theta_y =( theta' * X' )' - y;


J_noreg = 1/(2*m) *(ones(1, m) * (h_theta_y.^2));

theta_without_zero = theta(2:end, :);
n = size(theta_without_zero);
reg = lambda * ones(1,n) * (theta_without_zero.^2) / (2*m);
J = J_noreg + reg;


grad_noreg = (X' * h_theta_y).*(1/m);

grad_reg = theta .* (lambda / m);
grad_reg(1) = 0;
grad = grad_noreg + grad_reg;








% =========================================================================

grad = grad(:);

end
