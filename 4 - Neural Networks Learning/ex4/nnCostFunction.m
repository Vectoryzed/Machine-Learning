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

% PART 1

eye_matrix = eye(num_labels);
y_matrix = eye_matrix(y,:);

% Add ones to the a1 (X) input matrix
X = [ones(m, 1) X];
Z2 = (Theta1)*(X');
A2 = sigmoid(Z2);

A2 = A2';

% Add ones to the A2 matrix
A2 = [ones(size(A2,1), 1) A2];
Z3 = (Theta2)*(A2');
A3 = sigmoid(Z3);

A3 = A3';

% Cost function
J_1 = (1/m)*sum((sum( ((-y_matrix).*(log(A3))) - (1 - y_matrix).*log((1-A3)))));

% Cost function regularized
R_fact = (lambda)/(2*m);
temp_Theta1 = Theta1(:,2:end);
temp_Theta2 = Theta2(:,2:end);

J = J_1 + (R_fact)*(sum(sum(temp_Theta1.^2)) + sum(sum(temp_Theta2.^2)));

% Backpropagation
d3 = A3 - y_matrix;
d2 = ((d3) * (temp_Theta2)) .* sigmoidGradient(Z2');

Delta_1 = (d2') * X;
Delta_2 = (d3') * A2;

Theta1_grad = (Delta_1)/m;
Theta2_grad = (Delta_2)/m;

% -------------------------------------------------------------

% =========================================================================


%Regularized neural network
Theta1(:,1) = 0;
Theta2(:,1) = 0;

Theta1 = ((lambda)/(m))*(Theta1);
Theta2 = ((lambda)/(m))*(Theta2);

Theta1_grad = Theta1_grad + Theta1;
Theta2_grad = Theta2_grad + Theta2;

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];

end