function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples
n = length(theta); % number of parameter

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta


%% ====================== Cost Function ======================
hypothesis = sigmoid(X * theta);
h1 = -y .* log(hypothesis);
h2 = (1 - y) .* log(1 - hypothesis);

% Formula for not regularized cost function
NotRegularizedJ = (1/m) * sum(h1 - h2);

% Formula for regularized cost function. Note theta0 not involved in reglarization
theta0ExcludedTheta = theta(2:end,1);
J = NotRegularizedJ + (lambda / (2*m)) * sum(theta0ExcludedTheta.^2);

%% =============== Compute Cost and Gradient =================

grad(1) = ((1/m) .* sum( (hypothesis - y) .* X(:,1) ));

for i = 2:length(theta)
    grad(i) =( (1/m) .* sum( (hypothesis - y) .* X(:,i) ) ) + ( (lambda/m) * theta(i));
end


% =============================================================

end
