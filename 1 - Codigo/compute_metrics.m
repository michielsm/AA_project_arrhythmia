function [accuracy_total, auc,true_positives, false_positives, true_negatives, false_negatives,true_positives_mean,false_positives_mean,true_negatives_mean,false_negatives_mean, precision, recall,precision_mean, recall_mean, accuracy] =  compute_metrics(predicted_labels, Y_test) 

number_classes = rows(unique(Y_test))


m = rows(Y_test);
correctos = predicted_labels ==Y_test;
correctos = sum(correctos);
accuracy_total = (correctos/m)*100;


auc = zeros(number_classes, 1);
true_positives = zeros(number_classes, 1);
false_positives = zeros(number_classes, 1);
true_negatives = zeros(number_classes, 1);
false_negatives = zeros(number_classes, 1);
accuracy = zeros(number_classes, 1);
precision = zeros(number_classes, 1);
recall = zeros(number_classes, 1);



for a=1:number_classes
  a_class = find(a==Y_test);
  number_instances_a_class = rows(a_class);          

  ind_not_this_class = find(a!=Y_test);
  Y_tmp = Y_test;
  Y_tmp(ind_not_this_class) = -1;
  Y_tmp(a_class) = 1;
  
  
  predicted_a_class = find(a==predicted_labels);
  predicted_not_this_class = find(a!=predicted_labels);
  predicted_tmp = predicted_labels;
  predicted_tmp(predicted_a_class) = 1;
  predicted_tmp(predicted_not_this_class) = -1;
  
  
  auc(a) = auc_compute(Y_tmp, predicted_tmp, 1);

  true_positives(a) = sum((predicted_tmp==1)&(Y_tmp == 1));
  false_positives(a) = sum((predicted_tmp==1)&(Y_tmp != 1));

  true_negatives(a) = sum((predicted_tmp!=1)&(Y_tmp != 1));
  false_negatives(a) = sum((predicted_tmp!=1)&(Y_tmp == 1));
  
  precision(a) = true_positives(a) / (true_positives(a) + false_positives(a));
  recall(a) = true_positives(a) / (true_positives(a) + false_negatives(a));
  
  num_casos = rows(Y_tmp);
  correctos = predicted_tmp == Y_tmp;
  correctos = sum(correctos);
  accuracy(a) = (correctos/num_casos)*100;
endfor;

#METRICAS:====================================================================
auc
true_positives
false_positives
true_negatives
false_negatives
precision
recall

auc_mean = mean(auc(~isnan(auc)))

true_positives_mean = mean(true_positives(~isnan(true_positives)))
false_positives_mean = mean(false_positives(~isnan(false_positives)))
true_negatives_mean = mean(true_negatives(~isnan(true_negatives)))
false_negatives_mean = mean(false_negatives(~isnan(false_negatives)))

precision_mean = mean(precision(~isnan(precision)))
recall_mean = mean(recall(~isnan(recall)))

accuracy_total


endfunction