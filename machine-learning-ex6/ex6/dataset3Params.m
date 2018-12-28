function [C, sigma] = dataset3Params(X, y, Xval, yval)
%DATASET3PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = DATASET3PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C = 1;
sigma = 0.3;

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example, 
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using 
%        mean(double(predictions ~= yval))
%

[a,b]=size(X);
[c,d]=size(y);
[e,f]=size(Xval);
[g,h]=size(yval);

fprintf(" X=%d,%d y=%d,%d Xval=%d,%d yval=%d,%d \n",a,b,c,d,e,f,g,h);

a=zeros(4,1); 

for i=1:4

C=0.03*10^(i-1) ;
sigma=0.01*10^(i-1) ;

fprintf(" C=%g sigma=%g \n",C,sigma);

model= svmTrain(X, y, C, @(x1, x2) gaussianKernel(x1, x2, sigma));

predictions=svmPredict(model, Xval);

a(i)=mean(double(predictions ~= yval)) ;

end 

a 

[value,pos]=min(a); 

C=0.03*10^(pos-1) ;
sigma=0.01*10^(pos-1) ;

fprintf(" C=%g sigma=%g \n",C,sigma);

% =========================================================================

end
