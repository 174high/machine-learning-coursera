function p = predict(Theta1, Theta2, X)
%PREDICT Predict the label of an input given a trained neural network
%   p = PREDICT(Theta1, Theta2, X) outputs the predicted label of X given the
%   trained weights of a neural network (Theta1, Theta2)

% Useful values
m = size(X, 1);
num_labels = size(Theta2, 1);

% You need to return the following variables correctly 
p = zeros(size(X, 1), 1);

% ====================== YOUR CODE HERE ======================
% Instructions: Complete the following code to make predictions using
%               your learned neural network. You should set p to a 
%               vector containing labels between 1 to num_labels.
%
% Hint: The max function might come in useful. In particular, the max
%       function can also return the index of the max element, for more
%       information see 'help max'. If your examples are in rows, then, you
%       can use max(A, [], 2) to obtain the max for each row.
%

X_t = [ones(m,1) X];

[i,j]=size(X_t); 
    
fprintf("%%%%X_t= i=%d ,j=%d /n",i,j);

[i,j]=size(Theta1);          
    
fprintf("%%%%T= i=%d ,j=%d /n",i,j);

prediction1=X_t*Theta1';

[i,j]=size(prediction1);

fprintf("%%%% i=%d ,j=%d /n",i,j);

result1=sigmoid(prediction1);

result1_t=[ones(m,1) result1];

prediction2=result1_t*Theta2'; 

result2=sigmoid(prediction2);

for i=1:m

    [x,y]=max(result2(i,:))
    p(i)=y;

end 






% =========================================================================


end
