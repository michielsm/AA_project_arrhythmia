function [J grad] = _costeRN(params_rn, num_entradas, num_ocultas, num_etiquetas, X, y, lambda)
  #{
  Obtención de las matrices Theta (se desenrollan), estaban enrolladas en 1 matriz columna
  que hacía la labor de un array, ya que no se pueden juntar las 2 matrices según vienen,
  porque las dimensiones no coinciden, por eso se juntaron todas en 1 columna
  #}
  Theta1 = reshape(params_rn(1:num_ocultas * ( num_entradas + 1)), num_ocultas, ( num_entradas + 1));
  Theta2 = reshape(params_rn((1 + (num_ocultas * ( num_entradas + 1))):end), num_etiquetas, (num_ocultas + 1));

  m = rows(X);
  Y=zeros(m,num_etiquetas);

  etiquetas = eye(num_etiquetas); #etiquetas es de (10x10) Cada fila tiene un 1 en la columna=fila
  for i=1:m
    Y(i,:) = etiquetas(y(i),:);  #Ejemplo de valores de y [1..10], 5=[0 0 0 0 1 0 0 0 0 0]
  endfor
  #PROPAGACIÓN HACIA DELANTE para computar el valor de h0(x(i)) para cada ejemplo i====
  #===================a1==============================================
  a1 = X;
  a0 = ones(rows(a1),1);
  a1 = [a0, a1]; #Se le añade el término a0=1 a la a
  #===================a2==============================================
  z2 = a1*Theta1'; 
  a2 = _sigmoid_function(z2);
  a0 = ones(rows(a2),1);
  a2 = [a0, a2]; #Se le añade el término a0=1 a la a
  #====================a3=============================================
  z3 = a2*Theta2'; 
  a3 = _sigmoid_function(z3);
  #===================================================================
  h0 = a3; #La salida de la última capa es h0 (nuestra estimación)
  
  #================Función de coste J(0)=========================================
  J = (1/m)*sum(sum((-Y).*log(h0) - (1-Y).*log(1-h0), 2)); 
  #El último 2 significa sum(matriz, dimension=2), es decir se suma por columnas(el resultado es columna por tanto)
  regularizacion_thetas = (lambda/(2*m))*(sum(sum(Theta1(:, 2:end).^2, 2)) + sum(sum(Theta2(:,2:end).^2, 2)));
  J = J + regularizacion_thetas;
  
  #PROPAGACIÓN HACIA ATRÁS (backpropagation) para computar el valor de h0(x(i)) para cada ejemplo i====
  #================Cálculo del gradiente=========================================
  #Cálculo del error(sigma) en cada nivel
  sigma3 = a3 - Y;
  sigma2 = (sigma3*Theta2 .* _sigmoid_function_derivative([ones(rows(z2),1), z2]))(:,2:end);
  #[ones(rows(z2),1), z2] añade el término bias(el que lleva un 1) a cada caso de entrada

  #El (:,2:end) es para eliminar el término bias, es decir el que lleva un 1 al principio de cada nivel
  #Se tiene que añadir porque Theta2 lo tenía, así se pueden multiplicar las matrices, aunque 
  #luego se tenga que quitar porque el error de la variable 0 no nos sirve(porque siempre tiene valor
  #1 ese nodo y no se acumularía bien?
  
  #Acumulacón(delta) del error en cada nivel
  delta_1= sigma2'*a1;
  delta_2= sigma3'*a2;
  
  #Regularización de Theta:
  grad_1_regularizado = zeros(size(Theta1)); #Se reserva el espacio de memoria
  grad_2_regularizado = zeros(size(Theta2)); #Se reserva el espacio de memoria
  grad_1_regularizado = delta_1./m + (lambda/m)*[zeros(rows(Theta1), 1) Theta1(:, 2:end)];
  grad_2_regularizado = delta_2./m + (lambda/m)*[zeros(rows(Theta2), 1) Theta2(:, 2:end)];

  grad = [grad_1_regularizado(:) ; grad_2_regularizado(:)];



endfunction;