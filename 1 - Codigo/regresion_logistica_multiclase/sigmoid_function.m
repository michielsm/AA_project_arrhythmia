function sigmoid = sigmoid_function(z)

if isscalar(z)
  sigmoid = (1/(1+exp(-z)));
else isvector(z)
  sigmoid = (1./(1.+exp(-z)));
endif

endfunction;