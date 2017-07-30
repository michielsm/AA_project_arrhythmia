function _1_preprocess_data_binary_classes(all_features)


#0.1. Read and parse the data.
#==========================================================================#
fid = fopen("arrhythmia.data");
line = 0;
line_i=1; 
while (-1 ~= (line=fgetl(fid))) 
 elems = strsplit(line, ",");
 nums = str2double(elems);
 data(line_i++,:)= nums;
end 
fclose(fid); 

#0.2. Randomize the rows of the data.
#==========================================================================#
#data = _randomize_rows_data(data);


#0.3. Get X and Y from data.
#==========================================================================#
X = data(:, 1:columns(data)-1); #All columns except the 1st one
Y = data(:, columns(data):columns(data)); #Only the last columns

#The information of the dataset said it has 16 classes, but actually 3 classes don't
#have any instances, those classes are: 11, 12, 13. So I set the class number of 
#14, 15, 16 to 11, 12, 13, so this way we don't have any gap between real classes. 
for i=13:16
  class_to_substitute = find(i==Y);
  Y(class_to_substitute) = i-3;
endfor;

not_healthy = find(Y!=1);
Y(not_healthy) = 2;

number_instances = rows(X);
number_features = columns(X);
number_classes = 2;


mean_feature = zeros(number_features, 1);


#0.4. Feature selection: based in real life medical procedures:
#==========================================================================#
#{
En la vida real los ECG se miden normalmente con 12 o 13 canales(cables), como en este
dataset (12 canales). Sin embargo al realizar los holter(ECG portatil durante al menos 24 horas), 
suele ser bastantes menos canales, en mi caso la ultima vez solo fueron 3.

En nuestro TFG utilizamos algo intermedio: 8 canales, dado que su uso podra ser como
holter o como simplemente un ecg comun de una clinica para medir durante unos segundos.

Por ese motivo la seleccion de features la realizare con 8 canales. Descartamos por tanto
los canales DI, DII, DIII (por ser los cables que se ponen en las extremidades y no es
habitual ponerlos) y el V6 por estar muy cerca del V5 y no perder asi casi informacion.

     Of channel DI: 
      16 .. 27
      160 .. 169
     Of channel DII: 
      28 .. 39
      170 .. 179
     Of channels DIII:
      40 .. 51
      180 .. 189
     Of channel AVR:
      52 .. 63
      190 .. 199
     Of channel AVL:
      64 .. 75
      200 .. 209
     Of channel AVF:
      76 .. 87
      210 .. 219
     Of channel V1:
      88 .. 99
      220 .. 229
     Of channel V2:
      100 .. 111
      230 .. 239
     Of channel V3:
      112 .. 123
      240 .. 249
     Of channel V4:
      124 .. 135
      250 .. 259
     Of channel V5:
      136 .. 147
      260 .. 269
     Of channel V6:
      148 .. 159
      270 .. 279
#}

features_to_delete = [];
DI = [16:1:27];
DI = [DI, [160:1:169]];

DII = [28:1:39];
DII = [DII, [170:1:179]];

DIII = [40:1:51];
DIII = [DIII, [180:1:189]];

V6 = [148:1:159];
V6 = [V6, [270:1:279]];

features_to_delete = [DI, DII, DIII, V6];


if (exist("all_features", "var"))
  if (!all_features)
    X(:, features_to_delete) = []; #Delete the columns previously chosen
  endif;
else
  X(:, features_to_delete) = []; #Delete the columns previously chosen
endif;

#Recompute the rows and columns
number_instances = rows(X);
number_features = columns(X);

#1.1 Delete the features that are invariant (i.e. minimum = maximum in all the rows)
#==========================================================================#
columns_to_delete = [];
for i=1:number_features
  min_of_feature = min(X(:, i));
  max_of_feature = max(X(:, i));
  if min_of_feature==max_of_feature #Delete that feature (i.e that column)
    columns_to_delete = [columns_to_delete, i];
  endif;
  mean_feature(i) = mean(X(:, i)(~isnan(X(:, i))));
endfor;


#2. Set values in the features that are unknown (?):
#==========================================================================#
#We did the mean of all the instances for that feature and now we set the unknowns
#values to that mean, because we don't want to delete all the instance
#because just a few features of that instance are unknown.
for i=1:number_instances
  indices = isnan(X(i, :));
  indices = find(indices==1);
  for a=1:columns(indices) #If indices is empty, it won't enters here
    X(i, indices(a)) = mean_feature(indices(a));
  endfor;
endfor;

#1.2 Delete the features that are invariant (i.e. minimum = maximum in all the rows)
#==========================================================================#
#We delete them now, and not before because we needed all the means of the features
#and if we deleted these columns now, the indices would have disarrange
X(:, columns_to_delete) = []; #Delete the columns previously chosen
#Recompute the rows and columns
number_instances = rows(X);
number_features = columns(X);

#3. Normalization of the data:
#==========================================================================#
[X mu1 sigma1] = featureNormalize_firstTime(X);


#5. PCA (Principal component analysis):
#==========================================================================#
k = 2; #Number of PC(principal components) we want. We want 2 so we can plot the data
sigma = (1/number_features)*(X'*X);
[U, S, V] = svd(sigma);
Ureduce = U(:, 1:k);
z = X*Ureduce;

_plot_pca(z, Y);


#6. Set training, validation and test subsets:
#==========================================================================#
#{
I use 3 subsets: training, validation, and test.

---------------------------------------------------------
Training set + Validation set: are used to train the model.
---------------------------------------------------------
I train the model with the training set, and then I try their accuracy and AUC in 
the validation set(this is called cross validation). So I do that method several times
and the results of trying the model in the validation tests are useful to adjust the
parameter of each machine learning algorithm, so the validation test influences the model.

The method I use to perform the training influenced by a validation test is called:
--------------------------
k-fold cross validation
--------------------------
This method consists in partitioning the training set in 2 sets(trainig set and validation set)
The number of instances in each set is set by establishign a value: k.

k is the number of chunks the training set is divided. For example, if we have a training
set of 100 instances, if we set k=10, we'll have 10 chunks, and each one would have 10 values.

So we perform a loop of k(=10) iterations, and inside each iteration, following the previous example
we randomly divide the training set into training set(90 instances) and 
validation test(10 instances), and we train the model with the training test and test it in 
the validation test.

We do that loop n-times (we establish n to the value tdesired), and finally we get
what has been the best model and what were their parameters.




---------------------------------------------------------
Test set: it's used to test the accuracy and AUC of the model in a "real world" environment
---------------------------------------------------------
It's like a "real world" environment" because this instances were never used to influence 
the achieved model.


I use a split known as stratified, i.e it's not random but it stands for doing balanced
partition, so if we do 2 divisions(in fact we do 3(training+validation+test), this is just
a example), of 70% and 30%, and in total we have 10 instances of class 9, in the division we
would have 7 instances of class 9 in the 1st subset, and 3 instances of class 9 in the 2nd
subset.

I follow the recommendation a 60%(training set)-20%(validation set)-20%(test set).
#}

training_subtest_split_percentage = 0.6; 
validation_subtest_split_percentage = 0.2;
test_subtest_split_percentage = 0.2;

training_val_split_percentage = training_subtest_split_percentage + validation_subtest_split_percentage;

k_folds_cross_val = 4; #So, training+validation = 80%, and 80%/4 = 20% (validation test)

number_instances_training_subset = number_instances*training_subtest_split_percentage;
number_instances_validation_subset = number_instances*validation_subtest_split_percentage;
number_instances_test_subset = number_instances*test_subtest_split_percentage;

number_instances_training_and_val_subset = number_instances_training_subset + number_instances_validation_subset;


X_train_val = zeros(0, number_features);
X_test = zeros(0, number_features);
Y_train_val = zeros(0, 1);
Y_test = zeros(0, 1);
for i=1:number_classes
  i_class = find(i==Y);
  number_instances_i_class = rows(i_class);

  if number_instances_i_class > 2
    number_instances_i_class_train_val = round(number_instances_i_class*training_val_split_percentage);
    number_instances_i_class_test = round(number_instances_i_class*test_subtest_split_percentage);
  else
    number_instances_i_class_train_val = round(number_instances_i_class*0.5);
    number_instances_i_class_test = round(number_instances_i_class*0.5);
  endif
  i_class_train_val = i_class(1:number_instances_i_class_train_val);
  i_class_test = i_class(number_instances_i_class_train_val+1:number_instances_i_class);


  X_train_val = [X_train_val; X(i_class_train_val, :)];
  X_test = [X_test; X(i_class_test, :)];
  
  Y_train_val = [Y_train_val; Y(i_class_train_val, :)];
  Y_test = [Y_test; Y(i_class_test, :)];    



number_instances = rows(X);
number_features = columns(X);
    
      
endfor;

save -mat7-binary 'dataset_preprocessed_binary.mat', 'X_train_val', 'Y_train_val', 'X_test', 'Y_test', 'training_val_split_percentage', 'k_folds_cross_val';

endfunction
