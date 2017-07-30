function [J, grad] = lrCostFunction(theta, X, y, lambda)

m = length(y);
number_of_thetas = length(theta);

z = X*theta;
sigmoid = sigmoid_function(z);


J = ((1/m)*sum((-y.*log(sigmoid)) - ((1.-y).*log(1.-sigmoid)))) + ((lambda/(2*m))*(theta'*theta));


grad = ((1/m)*((sigmoid.-y)'*X)).+((lambda/m).*theta'); #A cada gradiente se le suma la carga de lambda
grad(1,1) = ((1/m)*((sigmoid(1,1).-y(1,1))'*X(1,1))); #Al grad(1,1) no se le aplica lambda
#=============================================================================
grad = grad'; #Se hace la traspuesta porque lo pide así la función fmincg

endfunction