function [porcentaje c sigma] = _svm_train_ecg(binary_classes)
more off;

#Selección de si se ha metido el parametro binary_classes(para que solo haya sanos-no_sanos
#o en su defecto que sean las 13 clases originales
if (exist("binary_classes", "var"))
  if (binary_classes)
    num_classes = 1; #We just need 1 loop(to compare class 1 vs class 2)
    labels = [1, 2];
    load("dataset_preprocessed_binary.mat"); #Guarda los datos de entrada en X e Y
  else
    num_classes = 13;
    labels = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13];
    load("dataset_preprocessed.mat"); #Guarda los datos de entrada en X e Y
  endif;
else
  num_classes = 13;
  labels = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13];
  load("dataset_preprocessed.mat"); #Guarda los datos de entrada en X e Y
endif

n=rows(Y_train_val);


k_fold_cross_validation = 4;





#c= [0.1:30:140];
#sigma = [0.001:0.001:0.003];
c= [0.1:2:105];
sigma = [0.0001:0.0001:0.002];

n=rows(Y_train_val);
c_done = zeros(length(c), 1);
sigma_done = zeros(length(sigma), 1);


aucs_mean = zeros(length(c), length(sigma));
aucs_one_fold = zeros(k_folds_cross_val, 1);

aucs_mean_training = zeros(length(c), length(sigma));
aucs_one_fold_training = zeros(k_folds_cross_val, 1);

for i=1:length(c)
  for j=1:length(sigma)
    for k_fold=1:k_folds_cross_val
      [X_train_fold Y_train_fold X_val_fold Y_val_fold] = split_fold(X_train_val, Y_train_val,training_val_split_percentage, k_folds_cross_val, k_fold);

      options_lib_svm = sprintf('-c %d -g %d -q', c(i), sigma(j));
      model = svmtrain(Y_train_fold, X_train_fold, options_lib_svm);
      [predicted_labels, accuracy, prob_estimates] = svmpredict(Y_val_fold, X_val_fold, model);
    
    
      for z=1:2000 #Contador de vueltas (Asi lo veo cuanto queda por la salida estandar)
        i
        j
      endfor
      
      auc = zeros(num_classes, 1);
      for a=1:num_classes
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
      
      
      #========Sobre training===========
      [predicted_labels, accuracy, prob_estimates] = svmpredict(Y_train_fold, X_train_fold, model);

      
      auc_training = zeros(num_classes, 1);
      for a=1:num_classes
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
    aucs_mean(i, j) = mean(aucs_one_fold(~isnan(aucs_one_fold)))
    aucs_mean_training(i, j) = mean(aucs_one_fold_training(~isnan(aucs_one_fold_training)))
        
    #sigma_done(i) =sigma(j);
  endfor
  #c_done(i) = c(i);
endfor

figure(2);
plot(c, mean(aucs_mean, 2), "b");
hold on;
plot(c,mean(aucs_mean_training, 2), "r");
xlabel("c", "fontsize", 11);
ylabel("AUC", "fontsize", 11);

figure(3);
plot(sigma, mean(aucs_mean, 1), "b");
hold on;
plot(sigma, mean(aucs_mean_training, 1), "r");
xlabel("sigma", "fontsize", 11);
ylabel("AUC", "fontsize", 11);


[best_model_i ind_best_model_i] = max(aucs_mean);
ind_best_model_i = ind_best_model_i(1);
[best_model ind_best_model_j] = max(best_model_i);

best_c = c(ind_best_model_i);
best_sigma = sigma(ind_best_model_j);

options_lib_svm = sprintf('-c %d -g %d -q', best_c, best_sigma);

if (exist("binary_classes", "var"))
  if (binary_classes)
      model = svmtrain(Y_train_val, X_train_val, options_lib_svm);
  else
    model = svmtrain(Y_train_val_oversampled, X_train_val_oversampled, options_lib_svm);
  endif;
  
else
    model = svmtrain(Y_train_val_oversampled, X_train_val_oversampled, options_lib_svm);
endif;

[predicted_labels, accuracy_total, prob_estimates] = svmpredict(Y_test, X_test, model)

 #========PLOT===========
    figure(1);
    number_features = columns(X_test);
    #PCA (Principal component analysis):
    k = 2; #Number of PC(principal components) we want. We want 2 so we can plot the data
    sigma = (1/number_features)*(X_test'*X_test);
    [U, S, V] = svd(sigma);
    Ureduce = U(:, 1:k);
    z = X_test*Ureduce;

    _plot_pca(z, Y_test);
    
    hold on;
    
    #----Plot boundary line-----------
    % calculate w and b
    w = model.SVs' * model.sv_coef;
    b = -model.rho;

    if model.Label(1) == 1
      w = -w;
      b = -b;
    end
    disp(w);
    disp(b);

    % plot the boundary line
    x = [min(z(:,1)):.01:max(z(:,1))];
    y = (-b - w(1)*x ) / w(2);
    hold on;
    plot(x,y)
    #Fuente del plot de la boundary line: http://www.alivelearn.net/wp-content/uploads/2009/10/cuixu_test_svm1.m
   #-------------------------------------
   #=========================


predicted_labels_svm = predicted_labels;
Y_test_svm = Y_test;
model_svm = model;
#To use the model: [predicted_labels, accuracy_total, prob_estimates] = svmpredict(Y_test, X_test, model)
save -mat7-binary 'dataset_model_svm.mat', 'predicted_labels_svm', 'Y_test_svm', 'model_svm';


best_c
best_sigma

[best_model_aucs_i ind_best_model_i1] = max(aucs_mean);
[best_model_aucs_mean ind_best_model_j] = max(best_model_aucs_i)

[best_model_aucs_t ind_best_model_i2] = max(aucs_mean_training);
[best_model_aucs_training ind_best_model_j] = max(best_model_aucs_t)
endfunction

#"one-against-one" multi-class method:
#==========================================================================#
#{
LIBSVM implements "one-against-one" multi-class method, so there are k(k-1)/2 binary models, 
where k is the number of classes.

We can consider two ways to conduct parameter selection.

For any two classes of data, a parameter selection procedure is conducted. 
Finally, each decision function has its own optimal parameters.
The same parameters are used for all k(k-1)/2 binary classification problems.
We select parameters that achieve the highest overall performance.

Each has its own advantages. A single parameter set may not be uniformly good for
all k(k-1)/2 decision functions. However, as the overall accuracy is the final consideration,
one parameter set for one decision function may lead to over-fitting. In the paper 

#}