function sigmoid = _sigmoid_function_derivative(z)

if isscalar(z)
	sigmoid = _sigmoid_function(z)*(1-_sigmoid_function(z));
elseif isvector(z)
	sigmoid = _sigmoid_function(z).*(1.-_sigmoid_function(z));
elseif ismatrix(z)
	sigmoid = _sigmoid_function(z).*(1.-_sigmoid_function(z));
endif

endfunction;