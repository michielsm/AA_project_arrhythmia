function [porcentaje_aciertos] = _red_neuronal_sinEntrenar_main(binary_classes)
more off;
#Selección de si se ha metido el parametro binary_classes(para que solo haya sanos-no_sanos
#o en su defecto que sean las 13 clases originales
if (exist("binary_classes", "var"))
  if (binary_classes)
    num_etiquetas = 2; 
    labels = [1, 2];
    load("dataset_preprocessed_binary.mat"); #Guarda los datos de entrada en X e Y
  else
    num_etiquetas = 13;
    labels = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13];
    load("dataset_preprocessed.mat"); #Guarda los datos de entrada en X e Y
  endif;
else
  num_etiquetas = 13;
  labels = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13];
  load("dataset_preprocessed.mat"); #Guarda los datos de entrada en X e Y
endif
#===========Carga de datos=======================================
m = rows(X_train_val); #Número ejemplos de entrenamiento

#{
#======Display de unos cuantos ejemplos de entrada================
% Selecciona aleatoriamente 100 ejemplos
rand_indices = randperm(m);
sel = X(rand_indices(1:100),:);

displayData(sel);
#}

#===========Inicialización aleatoria de Thetas===================
#Es necesario que sea aleatorio porque si no al calcular el gradiente
#el error seria siempre 0 para todos los nodos de la red
number_fixtures = columns(X_train_val);
num_filas_l1 = number_fixtures;
num_filas_l2 = 25;
num_filas_l3 = num_etiquetas;
%seleciona aleatoriamente 100 ejemplos
Theta1 = _pesosAleatorios(num_filas_l2, num_filas_l1+1)';
Theta2 = _pesosAleatorios(num_filas_l3, num_filas_l2+1)';

#Juntamos Theta1 y Theta2 en la misma matriz (hacemos una matriz columna con todas)
tam=rows(Theta1)*columns(Theta1);
Theta=reshape(Theta1,1,tam);
tam=rows(Theta2)*columns(Theta2);
Theta=[Theta reshape(Theta2,1,tam)];

num_entradas=number_fixtures;	
num_ocultas=num_filas_l2;		

#lambda = [0:0.15:6];
#num_nodos_capa_oculta = [num_ocultas:1:num_ocultas*2];
lambda = [0:0.15:1];
num_nodos_capa_oculta = [num_ocultas:6:num_ocultas*2];

aucs_mean = zeros(length(lambda), length(num_nodos_capa_oculta));
aucs_one_fold = zeros(k_folds_cross_val, 1);


aucs_mean_training = zeros(length(lambda), length(num_nodos_capa_oculta));
aucs_one_fold_training = zeros(k_folds_cross_val, 1);


for i=1:length(lambda)
  for j=1:columns(num_nodos_capa_oculta)
    j_num_ocultas = num_nodos_capa_oculta(j);
    for k_fold=1:k_folds_cross_val
      [X_train_fold Y_train_fold X_val_fold Y_val_fold] = split_fold(X_train_val, Y_train_val,training_val_split_percentage, k_folds_cross_val, k_fold);
    
      %seleciona aleatoriamente 100 ejemplos
      Theta1 = _pesosAleatorios(j_num_ocultas, num_filas_l1+1)';
      Theta2 = _pesosAleatorios(num_filas_l3, j_num_ocultas+1)';

      #Juntamos Theta1 y Theta2 en la misma matriz (hacemos una matriz columna con todas)
      tam=rows(Theta1)*columns(Theta1);
      Theta=reshape(Theta1,1,tam);
      tam=rows(Theta2)*columns(Theta2);
      Theta=[Theta reshape(Theta2,1,tam)];
    
    
      options = optimset('GradObj','on','MaxIter',600);  
      #checkNNGradients(lambda); #Se comparara nuestro gradiente obtenido(en costeRN) con el numérico estimado
      #Se miniminza J en la función fmincg, ya que le pasamos thet
      params_rn = fmincg(@(thetas)(_costeRN(thetas, num_entradas, j_num_ocultas, num_etiquetas, X_train_fold, Y_train_fold, lambda(i))),Theta' , options ) ;
      
      #Desenrollamos la matriz Theta que nos devulve fmincg en Theta1 y Theta2
      Theta1 = reshape(params_rn(1:j_num_ocultas * ( num_entradas + 1)), j_num_ocultas, ( num_entradas + 1));
      Theta2 = reshape(params_rn((1 + (j_num_ocultas * ( num_entradas + 1))):end), num_etiquetas, (j_num_ocultas + 1));
     
      #PROPAGACIÓN HACIA DELANTE para computar el valor de h0(x(i)) para cada ejemplo i====
      #===================a1==============================================
      a1 = X_val_fold;
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

      #=====Cálculo del porcentaje de casos que hemos acertado=====
      [maximo indicesMax] = max(h0'); 
      correctos = indicesMax'==Y_val_fold;
      correctos = sum(correctos);
      porcentaje_aciertos = (correctos/m)*100
      #===================================================================
      #En indicesMax se guarda la posicion en la que está el valor máximo de cada
      #ejemplo. Es decir es a nivel fila. Si el índice es 5 por ejemplo, querrá
      #decir que la posicion máxima era el 5 para ese ejemplo, por lo que nuestra red neuronal
      #habrá estimado que el valor para esa imagen es un 5    
    
      predicted_labels = indicesMax';
      auc = zeros(num_etiquetas, 1);
      for a=1:num_etiquetas
        a_class = find(a==Y_val_fold);
        number_instances_a_class = rows(a_class);          

        ind_not_this_class = find(a!=Y_val_fold);
        Y_tmp = Y_val_fold;
        Y_tmp(ind_not_this_class) = -1;
        Y_tmp(a_class) = 1;
        
        
        predicted_a_class = find(a==predicted_labels);
        predicted_not_this_class = find(a!=predicted_labels);
        predicted_tmp = predicted_labels;
        predicted_tmp(predicted_a_class) = 1;
        predicted_tmp(predicted_not_this_class) = -1;
        
        
        auc(a) = auc_compute(Y_tmp, predicted_tmp, 1)
       endfor;
       
      auc_mean = mean(auc(~isnan(auc)));
      aucs_one_fold(k_fold) = auc_mean
      
      #===========Sobre training=================
      #PROPAGACIÓN HACIA DELANTE para computar el valor de h0(x(i)) para cada ejemplo i====
      #===================a1==============================================
      a1 = X_train_fold;
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

      #=====Cálculo del porcentaje de casos que hemos acertado=====
      [maximo indicesMax] = max(h0'); 
      correctos = indicesMax'==Y_train_fold;
      correctos = sum(correctos);
      porcentaje_aciertos = (correctos/m)*100
      #===================================================================
      #En indicesMax se guarda la posicion en la que está el valor máximo de cada
      #ejemplo. Es decir es a nivel fila. Si el índice es 5 por ejemplo, querrá
      #decir que la posicion máxima era el 5 para ese ejemplo, por lo que nuestra red neuronal
      #habrá estimado que el valor para esa imagen es un 5    
    
      predicted_labels = indicesMax';
      auc_training = zeros(num_etiquetas, 1);
      for a=1:num_etiquetas
        a_class = find(a==Y_train_fold);
        number_instances_a_class = rows(a_class);          

        ind_not_this_class = find(a!=Y_train_fold);
        Y_tmp = Y_train_fold;
        Y_tmp(ind_not_this_class) = -1;
        Y_tmp(a_class) = 1;
        
        
        predicted_a_class = find(a==predicted_labels);
        predicted_not_this_class = find(a!=predicted_labels);
        predicted_tmp = predicted_labels;
        predicted_tmp(predicted_a_class) = 1;
        predicted_tmp(predicted_not_this_class) = -1;
        
        
        auc_training(a) = auc_compute(Y_tmp, predicted_tmp, 1)
       endfor;
       
      auc_mean_training = mean(auc_training(~isnan(auc_training)));
      aucs_one_fold_training(k_fold) = auc_mean_training
    
    endfor;
    aucs_mean(i, j) = mean(aucs_one_fold(~isnan(aucs_one_fold)));
    aucs_mean_training(i, j) = mean(aucs_one_fold_training(~isnan(aucs_one_fold_training)));

    
      for zz=1:2000
        i
        j
      endfor;
      
  endfor;
  

  
endfor;
  
figure(5);
plot(lambda, mean(aucs_mean, 2), "b");
hold on;
plot(lambda, mean(aucs_mean_training, 2), "r");
xlabel("lambda", "fontsize", 11);
ylabel("AUC", "fontsize", 11);



figure(6);
plot(num_nodos_capa_oculta, mean(aucs_mean, 1), "b");
hold on;
plot(num_nodos_capa_oculta, mean(aucs_mean_training, 1), "r");
xlabel("num_nodos_capa_oculta", "fontsize", 11);
ylabel("AUC", "fontsize", 11);
#========Cálculo de qué etiqueta hemos estimado que es cada ejemplo==============
num_casos = rows(X_test);

[best_model_i ind_best_model_i] = max(aucs_mean);
ind_best_model_i = ind_best_model_i(1);
[best_model ind_best_model_j] = max(best_model_i);

best_lambda = lambda(ind_best_model_i)
best_num_nodos_capa_oculta = num_nodos_capa_oculta(ind_best_model_j)


%seleciona aleatoriamente 100 ejemplos
Theta1 = _pesosAleatorios(best_num_nodos_capa_oculta, num_filas_l1+1)';
Theta2 = _pesosAleatorios(num_filas_l3, best_num_nodos_capa_oculta+1)';

#Juntamos Theta1 y Theta2 en la misma matriz (hacemos una matriz columna con todas)
tam=rows(Theta1)*columns(Theta1);
Theta=reshape(Theta1,1,tam);
tam=rows(Theta2)*columns(Theta2);
Theta=[Theta reshape(Theta2,1,tam)];

options = optimset('GradObj','on','MaxIter',600);  
#checkNNGradients(lambda); #Se comparara nuestro gradiente obtenido(en costeRN) con el numérico estimado
#Se miniminza J en la función fmincg, ya que le pasamos thet
if (exist("binary_classes", "var"))
  if (binary_classes)
    params_rn = fmincg(@(thetas)(_costeRN(thetas, num_entradas, best_num_nodos_capa_oculta, num_etiquetas, X_train_val, Y_train_val, best_lambda)),Theta' , options ) ;
  else
    params_rn = fmincg(@(thetas)(_costeRN(thetas, num_entradas, best_num_nodos_capa_oculta, num_etiquetas, X_train_val_oversampled, Y_train_val_oversampled, best_lambda)),Theta' , options ) ;
  endif;
else
  params_rn = fmincg(@(thetas)(_costeRN(thetas, num_entradas, best_num_nodos_capa_oculta, num_etiquetas, X_train_val_oversampled, Y_train_val_oversampled, best_lambda)),Theta' , options ) ;
endif



#Desenrollamos la matriz Theta que nos devulve fmincg en Theta1 y Theta2
Theta1 = reshape(params_rn(1:best_num_nodos_capa_oculta * ( num_entradas + 1)), best_num_nodos_capa_oculta, ( num_entradas + 1));
Theta2 = reshape(params_rn((1 + (best_num_nodos_capa_oculta * ( num_entradas + 1))):end), num_etiquetas, (best_num_nodos_capa_oculta + 1));

#PROPAGACIÓN HACIA DELANTE para computar el valor de h0(x(i)) para cada ejemplo i====
#===================a1==============================================
a1 = X_test;
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

#=====Cálculo del porcentaje de casos que hemos acertado=====
[maximo indicesMax] = max(h0'); 
correctos = indicesMax'==Y_test;
correctos = sum(correctos);
porcentaje_aciertos = (correctos/m)*100
#===================================================================
#En indicesMax se guarda la posicion en la que está el valor máximo de cada
#ejemplo. Es decir es a nivel fila. Si el índice es 5 por ejemplo, querrá
#decir que la posicion máxima era el 5 para ese ejemplo, por lo que nuestra red neuronal
#habrá estimado que el valor para esa imagen es un 5    

predicted_labels = indicesMax';

predicted_labels_neural = predicted_labels;
Y_test_neural = Y_test;
model_neural_theta1 = Theta1;
model_neural_theta2 = Theta2;
save -mat7-binary 'dataset_model_neural.mat', 'predicted_labels_neural', 'Y_test_neural', 'model_neural_theta1', 'model_neural_theta2';


best_lambda
best_num_nodos_capa_oculta

[best_model_aucs_mean ind_best_model_i] = max(aucs_mean)
[best_model_aucs_mean_training ind_best_model_it] = max(aucs_mean_training)

endfunction
  