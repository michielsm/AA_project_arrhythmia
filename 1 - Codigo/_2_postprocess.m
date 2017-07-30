function _2_postprocess()

load("dataset_model_reg.mat");
load("dataset_model_neural.mat");
load("dataset_model_svm.mat");


#POST-PROCESSING===============================

#=============REGRESION==================================
display("")
display("==================")
display("==========REGRESION========")
display("==================")
[accuracy_total, auc,true_positives, false_positives, true_negatives, false_negatives,true_positives_mean,false_positives_mean,true_negatives_mean,false_negatives_mean, precision, recall,precision_mean, recall_mean, accuracy] =  compute_metrics(predicted_labels_reg, Y_test_reg);
#=============REDES NEURONALES==================================
display("")
display("==================")
display("==========REDES NEURONALES========")
display("==================")
[accuracy_total, auc,true_positives, false_positives, true_negatives, false_negatives,true_positives_mean,false_positives_mean,true_negatives_mean,false_negatives_mean, precision, recall,precision_mean, recall_mean, accuracy] =  compute_metrics(predicted_labels_neural, Y_test_neural); 


#=============SVM==================================
display("")
display("==================")
display("==========SVM========")
display("==================")

[accuracy_total, auc,true_positives, false_positives, true_negatives, false_negatives,true_positives_mean,false_positives_mean,true_negatives_mean,false_negatives_mean, precision, recall,precision_mean, recall_mean, accuracy] =  compute_metrics(predicted_labels_svm, Y_test_svm);


#1. True positive rate:
#==========================================================================#


#2. False positive rate:
#==========================================================================#


#3. AUC (combination of false positives and negatives, and probability of have a false positive/negative, i.e accuracy of the algorithm):
	#i.e. Percentages of success and error
#==========================================================================#
#



endfunction
