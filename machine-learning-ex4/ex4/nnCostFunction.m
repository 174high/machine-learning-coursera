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

%[a,b]=size(nn_params);
%fprintf("size of nn_param =%d,%d  \n",a,b);

Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

[a,b]=size(Theta1);
[c,d]=size(Theta2); 
fprintf("size of theta1 =%d,%d size of theta2=%d,%d \n",a,b,c,d); 


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

X=[ones(m,1 ) X];
[a,b]=size(X);
[c,d]=size(Theta1);
fprintf("size of X= %d,%d theta=%d,%d \n",a,b,c,d);

prediction=sigmoid(X*Theta1') ;

[a,b]=size(prediction);
fprintf("size of prediciton=%d,%d \n",a,b);

prediction=[ones(m,1) prediction];

prediction2=sigmoid(prediction*Theta2'); 

[a,b]=size(prediction2);
fprintf("size of prediciton2=%d,%d \n",a,b);


% -ylog(prediction2)-(1-y)log(1-prediction2)

for i=1:1

for j=1:1

%prediction2(i,j)

end 

end 

sum1=0 ;

%fprintf(" m=%d,num_labels=%d \n",m,num_labels);

for i=1:m 

for j=1:num_labels

%        fprintf("prediction2(%d,%d)=%g y(%d)=%d \n",i,j,prediction2(i,j),i,y(i));
    	
	if j==y(i)
	sum1=sum1-log(prediction2(i,j));
	else
	sum1=sum1-log(1-prediction2(i,j)); 
        end 

end 

end 

%fprintf(" input=%d,hidden=%d \n",input_layer_size,hidden_layer_size);

sum2=0 ; 

for i=1:hidden_layer_size 
	sum2=sum2+sum(Theta1(i,:).^2);
end 

sum2=sum2-sum(Theta1(:,1).^2); 


sum3=0 ; 

for i=1:num_labels
        sum3=sum3+sum(Theta2(i,:).^2);
end 

sum3=sum3-sum(Theta2(:,1).^2);


J=sum1/m+(lambda/(2*m))*(sum2+sum3) ; 

%J=sum1/m; 


fprintf(" J=%g \n",J);

%------------------------------------------------------

%for i=1:m 

[a,b]=size(X);
[c,d]=size(Theta1);
%fprintf("size of X= %d,%d theta=%d,%d \n",a,b,c,d);

z2=X*Theta1'; 

a2=sigmoid(z2) ;

[a,b]=size(z2);
[c,d]=size(a2);
%fprintf("size of z2= %d,%d a2=%d,%d \n",a,b,c,d);

a2=[ones(m,1) a2]; 

[c,d]=size(a2);
fprintf("size of z2= %d,%d a2=%d,%d \n",a,b,c,d);

z3=a2*Theta2' ;

a3=sigmoid(z3);

[a,b]=size(z3);
[c,d]=size(a3);
[e,f]=size(y); 
[g,h]=size(Theta2);
%fprintf("size of z3= %d,%d a3=%d,%d y=%d,%d theta=%d,%d \n",a,b,c,d,e,f,g,h);

tmp_error=zeros(num_labels); 

for i=1:m

tmp_error=((1:4)==y(i)) ; 

error3=a3(i)-tmp_error; 

tmp=error3*Theta2;

tmp_a2=a2(i,:).*(1-a2(i,:))

alpha=tmp.*tmp_a2

alpha=alpha(:,2:end);

[a,b]=size(Theta2);
[c,d]=size(error3);
[e,f]=size(tmp);
[g,h]=size(a2(i,:)); 
[i,j]=size(tmp_a2);
[k,l]=size(alpha);
fprintf(" Thate2=%d,%d:error3=%d,%d tmp=%d,%d a2(%d)=%d,%d tmp_a2=%d,%d alpha=%d,%d \n",a,b,c,d,e,f,i,g,h,i,j,k,l);




end 




%end 



% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
