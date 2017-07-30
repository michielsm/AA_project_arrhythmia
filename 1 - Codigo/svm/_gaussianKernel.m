function [similarity] = _gaussianKernel (x1 , x2 , sigma )

similarity = exp(-(sum((x1.-x2).^2)/(2*(sigma^2))))

endfunction
