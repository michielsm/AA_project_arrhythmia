function [thetas] = oneVsAll(X, y, num_etiquetas, lambda)
%ONEVSALL entrena varios clasificadores por regresión logística y devuelve
% el resultado en una matriz all_theta , donde l a fila iésima
% corresponde al clasificador de la etiqueta i!ésima
num_fixtures = columns(X);

initial_theta = zeros(num_fixtures, 1);

options=optimset('GradObj','on','MaxIter', 600);
for i=1:num_etiquetas
  #Con el y==c de la siguiente llamada, conseguimos hacer el OneVsAll
  #es decir ponemos a 1 la etiqueta que queramos entrenar, y a 0 todas las demás
  thetas(i, :) = _fmincg(@(t)(lrCostFunction(t, X,(y==i), lambda)), initial_theta, options);
endfor;


endfunction