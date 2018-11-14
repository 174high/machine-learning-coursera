function [J, grad] = costFunction(theta, X, y)
%COSTFUNCTION Compute cost and gradient for logistic regression
%   J = COSTFUNCTION(theta, X, y) computes the cost of using theta as the
%   parameter for logistic regression and the gradient of the cost
%   w.r.t. to the parameters.

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
%
% Note: grad should have the same dimensions as theta
%

prediction=X*theta;

[i,j]=size(prediction); 

fprintf("%%%% i=%d ,j=%d /n",i,j);

result=sigmoid(prediction);

fprintf("%%%% i=%d ,j=%d \n",i,j);

a1=log(result);
a2=log(1-result); 

b1=y.*a1 ;
b2=(1-y).*a2 ;

c1=-b1-b2 ;

J=sum(c1)/m;

#b1=a1*y; 

[a,b]=size(result-y); 

fprintf("%%%% a=%d ,b=%d \n",a,b);

grad=(X'*(result-y))/m ;


% =============================================================

end
