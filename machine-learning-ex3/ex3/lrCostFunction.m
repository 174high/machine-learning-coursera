function [J, grad] = lrCostFunction(theta, X, y, lambda)
%LRCOSTFUNCTION Compute cost and gradient for logistic regression with 
%regularization
%   J = LRCOSTFUNCTION(theta, X, y, lambda) computes the cost of using
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
%
% Hint: The computation of the cost function and gradients can be
%       efficiently vectorized. For example, consider the computation
%
%           sigmoid(X * theta)
%
%       Each row of the resulting matrix will contain the value of the
%       prediction for that example. You can make use of this to vectorize
%       the cost function and gradient computations. 
%
% Hint: When computing the gradient of the regularized cost function, 
%       there're many possible vectorized solutions, but one solution
%       looks like:
%           grad = (unregularized gradient for logistic regression)
%           temp = theta; 
%           temp(1) = 0;   % because we don't add anything for j = 0  
%           grad = grad + YOUR_CODE_HERE (using the temp variable)
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

theta_t=theta;

theta_t(1,:)=[] ;

J=sum(c1)/m+(lambda*sum(theta_t.*theta_t))/(2*m);

#b1=a1*y;

[a,b]=size(result-y);

fprintf("%%%% a=%d ,b=%d \n",a,b);


grad=(X'*(result-y))/m+(lambda/m)*theta;

%grad=(X'*(result-y))/m;

grad_t=(X'*(result-y))/m;

[a,b]=size(grad);

fprintf("&&&&& a=%d,b=%d &&&&\n ",a,b);

grad(1)=grad_t(1);





% =============================================================

grad = grad(:);

end
