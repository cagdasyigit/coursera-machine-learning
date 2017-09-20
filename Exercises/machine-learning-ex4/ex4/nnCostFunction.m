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

%--- Adding Bias To Input Unit ---
X = [ones(m, 1) X];

%--- Calculate Hidden Unit Activation Value
z2 = Theta1 * X';
a2 = sigmoid(z2);

%--- Adding Bias Unit To Hidden Activation Unit
a2 = [ones(m, 1) a2'];

%--- Calculate Output Unit Activation Value
z3 = Theta2 * a2';
a3 = sigmoid(z3);

%--- Unroll y vector to 10x10 matrice
yVec = zeros(m,num_labels);

for i = 1:m
    yVec(i,y(i)) = 1;
end

%--- Build Cost Function
h1 = -yVec .* log(a3)';
h2 = (1 - yVec) .* log(1 - a3)';
hyphotesis =  h1 - h2;
J = sum(sum(hyphotesis)) / m;

%--- We need to substract first weights because they are bias unit weights
Theta1R = Theta1(:, 2:end);
Theta2R = Theta2(:, 2:end);
regularization = (sum(sum(Theta1R .^ 2)) + sum(sum(Theta2R .^ 2))) * (lambda / (2 * m));

%--- Add Regularization
J = J + regularization;

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

for t = 1:m
  % For the input layer, where L=1:
	a1 = X(t,:); %--- 1x401 Matrice

	% For the hidden layers, where L=2:
	z2 = Theta1 * a1';
	a2 = [1; sigmoid(z2)]; %--- 10x1

	z3 = Theta2 * a2;
	a3 = sigmoid(z3); %--- 10x1
  
  %--- Check output classes and pick one which is equals one (true)
  yk = ([1:num_labels] == y(t)); %--- 1x10 Matrice
  
  %--- Output error difference
  d3 = a3 - yk'; %--- 10x1 Matrice
  
  %--- Hidden input difference
  d2 = Theta2' * d3 .* [1; sigmoidGradient(z2)];
  d2 = d2(2:end);
  
  %--- Calculate Gradients
  Theta1_grad = Theta1_grad + (d2 * a1); %--- 25x401 Matrice
  Theta2_grad = Theta2_grad + (d3 * a2');
end

Theta1_grad = Theta1_grad * (1/m);
Theta2_grad = Theta2_grad * (1/m);

% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%

Theta1_grad = Theta1_grad + (lambda / m) * [zeros(size(Theta1, 1), 1) Theta1(:,2:end)];
Theta2_grad = Theta2_grad + (lambda / m) * [zeros(size(Theta2, 1), 1) Theta2(:,2:end)];

% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
