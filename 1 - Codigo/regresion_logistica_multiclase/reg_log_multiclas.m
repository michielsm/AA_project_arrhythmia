function porcentajeCorrectos = reg_log_multiclas(binary_classes)
more off;

#Selección de si se ha metido el parametro binary_classes(para que solo haya sanos-no_sanos
#o en su defecto que sean las 13 clases originales
if (exist("binary_classes", "var"))
  if (binary_classes)
    num_etiquetas = 2; 
    etiquetas = [1 2];
    load("dataset_preprocessed_binary.mat"); #Guarda los datos de entrada en X e Y
  else
    num_etiquetas = 13;
    etiquetas = [1 2 3 4 5 6 7 8 9 10 11 12 13];
    load("dataset_preprocessed.mat"); #Guarda los datos de entrada en X e Y
  endif;
else
  num_etiquetas = 13;
  etiquetas = [1 2 3 4 5 6 7 8 9 10 11 12 13];
   load("dataset_preprocessed.mat"); #Guarda los datos de entrada en X e Y
endif

#lambda = [0:0.025:3];
lambda = [0:0.005:1];
n_iteraciones = length(lambda);
aucs_mean = zeros(length(lambda), 1);
aucs_one_fold = zeros(k_folds_cross_val, 1);

aucs_mean_training = zeros(length(lambda), 1);
aucs_one_fold_training = zeros(k_folds_cross_val, 1);

for i=1:length(lambda)
  for k_fold=1:k_folds_cross_val
      #Training:
      #==========================================================================#
      [X_train_fold Y_train_fold X_val_fold Y_val_fold] = split_fold(X_train_val, Y_train_val,training_val_split_percentage, k_folds_cross_val, k_fold);
      #===Se le añade el termino x0=1 a todos los casos de entrada========
      x0 = ones(rows(X_train_fold),1);
      X_train_fold = [x0, X_train_fold];
      #===================================================================
      
      thetas = oneVsAll(X_train_fold, Y_train_fold, num_etiquetas, lambda(i));
      
      #Cross validation:
      #==========================================================================#
      num_casos = rows(X_val_fold);
      #===Se le añade el termino x0=1 a todos los casos de entrada========
      x0 = ones(rows(X_val_fold),1);
      X_val_fold = [x0, X_val_fold];

      h0 = X_val_fold*thetas'; 
      [maximo indicesMax] = max(h0'); 
      correctos=indicesMax'==Y_val_fold;
      correctos = sum(correctos);
      porcentajeCorrectos = (correctos/num_casos)*100
      
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
      
    
    
    

    
    
        #==================Sobre training=====================================#
      num_casos = rows(X_train_fold);
      #===Se le añade el termino x0=1 a todos los casos de entrada========
      #x0 = ones(rows(X_train_fold),1);
      #X_train_fold = [x0, X_train_fold];

      h0 = X_train_fold*thetas'; 
      [maximo indicesMax] = max(h0'); 
      correctos=indicesMax'==Y_train_fold;
      correctos = sum(correctos);
      porcentajeCorrectos = (correctos/num_casos)*100
      
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
    
    aucs_mean(i) = mean(aucs_one_fold(~isnan(aucs_one_fold)))
    aucs_mean_training(i) = mean(aucs_one_fold_training(~isnan(aucs_one_fold_training)))
    
    for zz=1:1000 #Para ver por la salida estándar por donde va el algoritmo
      i
    endfor;

endfor;
#===================================================================

figure(4);
plot(lambda, aucs_mean);
hold on;
plot(lambda, aucs_mean_training, "r");
xlabel("lambda", "fontsize", 11);
ylabel("AUC", "fontsize", 11);

#========Cálculo de qué etiqueta hemos estimado que es cada ejemplo==============
num_casos = rows(X_test);

[best_model_I, ind_best_model_i] = max(aucs_mean);
best_lambda = lambda(ind_best_model_i);
#===Se le añade el termino x0=1 a todos los casos de entrada========
if (exist("binary_classes", "var"))
  if (binary_classes)
      x0 = ones(rows(X_train_val),1);
      X_train_val = [x0, X_train_val];

     thetas = oneVsAll(X_train_val, Y_train_val, num_etiquetas, best_lambda);
  else
    x0 = ones(rows(X_train_val_oversampled),1);
    X_train_val_oversampled = [x0, X_train_val_oversampled];

    thetas = oneVsAll(X_train_val_oversampled, Y_train_val_oversampled, num_etiquetas, best_lambda);
  endif;
  
else
    x0 = ones(rows(X_train_val_oversampled),1);
    X_train_val_oversampled = [x0, X_train_val_oversampled];

    thetas = oneVsAll(X_train_val_oversampled, Y_train_val_oversampled, num_etiquetas, best_lambda);
endif;


#===Se le añade el termino x0=1 a todos los casos de entrada========
x0 = ones(rows(X_test),1);
X_test = [x0, X_test];

h0 = X_test*thetas'; 
[maximo indicesMax] = max(h0'); 
correctos=indicesMax'==Y_test;
correctos = sum(correctos);
porcentajeCorrectos = (correctos/num_casos)*100
predicted_labels = indicesMax';

predicted_labels_reg = predicted_labels;
Y_test_reg = Y_test;
model_reg = thetas; 
#{
Se sacan predicted labels con h0 = X_test*thetas'; 
[maximo indicesMax] = max(h0'); 
predicted_labels = indicesMax';
#}
save -mat7-binary 'dataset_model_reg.mat', 'predicted_labels_reg', 'Y_test_reg', 'model_reg';

best_lambda
[best_aucs_mean, ind_best_model_i] = max(aucs_mean)
[best_aucs_mean_training, ind_best_model_it] = max(aucs_mean_training)

#En indicesMax se guarda la posicion en la que está el valor máximo de cada
#ejemplo. Es decir es a nivel fila. Si el índice es 5 por ejemplo, querrá
#decir que la posicion máxima era el 5 para ese ejemplo, por lo que nuestro programa
#habrá estimado que el valor para esa imagen es un 5

#h0: Para cada caso se saca cuantas probabilidades tiene de ser x etiqueta.
#Es decir h0 es un vector en el cual el número más alto corresponde a la 
#probabilidad de ser esa etiqueta, así que nos quedamos con esa.
#Ejemplo h0 = [13, 16, 95] nos quedaríamos con 95, ya que se podría decir que hay
#un 95% de probabilidad de que sea esa etiqueta (las etiquetas en este caso serían
# [1,2,3] por ejemplo. La conclusión sería que hemos estimado que ese caso es un 3
  
#Nota: Se podría haber hecho la función sigmoide sobre h0, pero no es necesario en este caso.
#Al no hacerla simplemente tenemos un rango de valores más allá de 0 a 1, pero no nos influye
#para sacar el máximo valor.
#===================================================================



endfunction