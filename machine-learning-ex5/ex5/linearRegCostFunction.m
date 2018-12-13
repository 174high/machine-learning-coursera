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



[a,b]=size(X); 
[c,d]=size(theta); 
[e,f]=size(y); 
[g,h]=size(X*theta); 
fprintf(" X=%d,%d theta=%d,%d y=%d,%d x*theta=%d,%d \n",a,b,c,d,e,f,g,h);

prediction=sum((X*theta-y).^2)  ;  

[a,b]=size(prediction); 

fprintf("prediction=%d,%d \n",a,b);

theta.^2 

sum(theta.^2) 

theta1=theta(2:end,:);


[a,b]=size(theta); 
[c,d]=size(theta1);
fprintf(" thata=%d,%d theta1=%d,%d \n",a,b,c,d);
J =prediction/(2*m)+((sum(theta1.^2))*lambda)/(2*m) ;   




grad=(X'*(X*theta-y))/m + (lambda*theta)/m;  

grad(1)=grad(1)- ((lambda*theta)/m)(1) ;

[a,b]=size(X*theta-y);
[c,d]=size(X');
[e,f]=size(grad);
fprintf(" t1=%d,%d \n",a,b);
fprintf(" t2=%d,%d \n",c,d);
fprintf(" t3=%d,%d \n",e,f); 

% =========================================================================

grad = grad(:);

end
