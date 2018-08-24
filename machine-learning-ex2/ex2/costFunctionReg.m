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
regsum = 0;
for j = 2:(size(theta))(1),
  regsum = regsum + theta(j,1) ^ 2;
end;
regsum = regsum * lambda / (2 * m);



sum = 0;
for i=1: m,
  hi = sigmoid((theta' * X')(i));
  yi = y(i);
  temp = -yi * log(hi) - (1-yi)*log(1-hi);
  sum = sum + temp;
end;
J = sum/m + regsum;

gradTemp = zeros(size(theta));

for j = 1:size(theta, 1),
    sum = 0;
    for i = 1:m,
      hi = sigmoid((theta' * X')(i));
      sum = sum + (hi - y(i))*X(i,j);
    end;
    if j == 1,
      tempGrad(j, 1) =  sum/m;
    else
      tempGrad(j, 1) = sum/m + lambda * theta(j, 1) / m;
    end;
    
  end;
  grad = tempGrad;

% =============================================================

end
