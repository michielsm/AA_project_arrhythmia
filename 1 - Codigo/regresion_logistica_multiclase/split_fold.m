function [X_train_fold Y_train_fold X_val_fold Y_val_fold] = split_fold(X_train_val, Y_train_val,training_val_split_percentage, k_folds_cross_val, k_fold)

number_features = columns(X_train_val);
number_instances = rows(X_train_val);
val_subtest_split_percentage = training_val_split_percentage / k_folds_cross_val; #0.8 / 4 = 0.2
training_subtest_split_percentage = training_val_split_percentage - val_subtest_split_percentage; #0.8 - 0.2 = 0.6

number_instances_training_subset = round(number_instances*training_subtest_split_percentage);
number_instances_validation_subset = round(number_instances*val_subtest_split_percentage);


rows_shuffled = randperm(rows(X_train_val));
X_train_val = X_train_val(rows_shuffled,:);
Y_train_val = Y_train_val(rows_shuffled,:);


X_train_fold = X_train_val(1:number_instances_training_subset, :);
X_val_fold = X_train_val(number_instances_training_subset+1:end, :);

Y_train_fold = Y_train_val(1:number_instances_training_subset, :);
Y_val_fold = Y_train_val(number_instances_training_subset+1:end, :);







number_classes = 13;
for i=1:number_classes
  i_class_train_val = find(i==Y_train_val);
  i_class_train_fold = find(i==Y_train_fold);
  i_class_val_fold = find(i==Y_val_fold);
  
    number_instances_i_class = rows(i_class_train_val);
    
    #Overfitting of classes impossible to detect without overfitting
  if (i==7 || i==8 || i == 11 || i ==12 || i ==13)  
      X_train_fold(i_class_train_fold,:) = [];
      X_val_fold(i_class_val_fold,:) = [];
      Y_train_fold(i_class_train_fold,:) = [];
      Y_val_fold(i_class_val_fold,:) = [];
      
      X_train_fold = [X_train_fold; X_train_val(i_class_train_val, :)];
      X_val_fold = [X_val_fold; X_train_val(i_class_train_val, :)];
    
      Y_train_fold = [Y_train_fold; Y_train_val(i_class_train_val, :)];
      Y_val_fold = [Y_val_fold; Y_train_val(i_class_train_val, :)];
   endif;
   
   
   
   
   #Oversampling of classes with very few instances
  if (number_instances_i_class < 10)
  #I duplicate the values(15 times: 1(original insert)+14 duplications)
  #to achieve the oversampling on these classes
      duplicate_times = 14;
  
      if (i != 7 && i!=8 && i != 11 && i != 12 && i != 13) 
        nuevas_filas_x_train_fold = X_train_fold(i_class_train_fold, :);
        nuevas_filas_y_train_fold = Y_train_fold(i_class_train_fold, :);
        X_train_fold(i_class_train_fold,:) = [];
        Y_train_fold(i_class_train_fold,:) = [];
        
       for (a=1:duplicate_times)
          X_train_fold = [X_train_fold; nuevas_filas_x_train_fold];
          Y_train_fold = [Y_train_fold; nuevas_filas_y_train_fold];
       endfor
     
     endif;
  endif
   
endfor;

endfunction